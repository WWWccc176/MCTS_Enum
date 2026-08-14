from __future__ import annotations

import time

import torch

from .config import NetworkConfig, RuntimeConfig, TrainConfig
from .inference_pool import MultiDeviceInferencePool
from .model import MCTSNet
from .trainer import Trainer


def _infer_apply(engine, pool: MultiDeviceInferencePool, requests: list[dict]) -> None:
    logits, values = pool.infer(requests)
    request_ids = [int(request["request_id"]) for request in requests]
    engine.submit_inference(request_ids, logits, values)


def _refresh_all_priors(
    engine,
    pool: MultiDeviceInferencePool,
    runtime: RuntimeConfig,
) -> None:
    cursor = 0
    while cursor < int(engine.node_count):
        payload = engine.collect_refresh_batch(cursor, runtime.refresh_batch_size)
        requests = list(payload["requests"])
        next_cursor = int(payload["next_cursor"])
        if next_cursor <= cursor:
            raise RuntimeError("global refresh cursor did not advance")
        if requests:
            logits, values = pool.infer(requests)
            node_ids = [int(request["node_id"]) for request in requests]
            engine.apply_refresh(node_ids, logits, values)
        cursor = next_cursor
    engine.mark_refresh_complete()


class NetController:
    def __init__(
        self,
        engine,
        device: torch.device,
        runtime: RuntimeConfig,
        train_config: TrainConfig,
        network_config: NetworkConfig,
    ):
        self.runtime = runtime
        self.train_config = train_config
        bootstrap = list(engine.collect_inference_batch(1))
        if not bootstrap:
            raise RuntimeError(
                "MCTS root did not produce an initial inference request: "
                + str(engine.diagnostic_status())
            )

        model = MCTSNet(
            global_dim=len(bootstrap[0]["global"]),
            recent_dim=len(bootstrap[0]["recent"]),
            config=network_config,
        ).to(device)
        model.eval()
        self.pool = MultiDeviceInferencePool(
            model, device, runtime.max_padded_candidates_per_device
        )
        self.trainer = Trainer(self.pool.primary_model, self.pool.primary_device, train_config)
        self.update_index = 0
        self.completed_nodes = 0
        self._set_phase_gso(engine)
        _infer_apply(engine, self.pool, bootstrap)

    @property
    def gpu_count(self) -> int:
        return self.pool.gpu_count

    def _set_phase_gso(self, engine) -> None:
        features = list(engine.gso_features())
        self.pool.set_gso_features(features)
        self.trainer.set_gso_features(features)

    def reset_replay_for_new_basis(self) -> None:
        self.trainer.reset_replay()

    def bootstrap_new_phase(self, engine) -> None:
        self._set_phase_gso(engine)
        bootstrap = list(engine.collect_inference_batch(1))
        if bootstrap:
            _infer_apply(engine, self.pool, bootstrap)
        elif not bool(engine.finished):
            raise RuntimeError(
                "new MCTS phase produced neither bootstrap inference nor a valid stop: "
                + str(engine.diagnostic_status())
            )


    def _status_line(self, engine, now: float, force: bool = False) -> None:
        interval = max(0.25, float(self.runtime.status_interval_seconds))
        if not force and now - self._last_status_time < interval:
            return
        status = dict(engine.status())
        nodes = int(status["nodes"])
        dt = max(now - self._last_status_time, 1.0e-9)
        rate = (nodes - self._last_status_nodes) / dt
        budget = int(status["node_budget"])
        budget_text = "inf" if budget < 0 else str(budget)
        pct = "" if budget < 0 else f" {100.0 * nodes / max(1, budget):5.1f}%"
        print(
            f"[net] phase={int(status['phase'])} "
            f"nodes={nodes}/{budget_text}{pct} "
            f"total={self.completed_nodes + nodes} rate={rate:,.0f}/s "
            f"pending={int(status['pending'])} "
            f"best/GH={float(status['best_quality_ratio']):.8f} "
            f"best@={int(status['best_found_at_node_count'])} "
            f"rootN={int(status['root_visits'])} nn={self.update_index} "
            f"refresh={int(status['nodes_since_refresh'])}/{self.runtime.refresh_interval} "
            f"Rdrop={int(status['radius_drops_since_global_update'])}/{self.runtime.radius_global_update_interval}",
            flush=True,
        )
        self._last_status_time = now
        self._last_status_nodes = nodes

    def run_phase(self, engine) -> None:
        self._last_status_time = time.monotonic()
        self._last_status_nodes = int(engine.node_count)
        self._status_line(engine, self._last_status_time, force=True)
        while not bool(engine.finished):
            progress_before = int(engine.progress_epoch)
            requests = list(
                engine.collect_inference_batch(self.runtime.inference_batch_size)
            )
            progress_after = int(engine.progress_epoch)

            if requests:
                _infer_apply(engine, self.pool, requests)
            elif bool(engine.finished):
                break
            elif progress_after != progress_before:
                # Exact terminal evaluation, pruning, or closure propagation can
                # advance the tree without requiring a neural-network call.
                pass
            else:
                raise RuntimeError(
                    "MCTS backend is unfinished but made no exact-search progress "
                    "and produced no inference request: "
                    + str(engine.diagnostic_status())
                )

            self._status_line(engine, time.monotonic())

            if int(engine.nodes_since_refresh) >= self.runtime.refresh_interval:
                samples = list(
                    engine.training_samples(
                        self.train_config.max_training_samples_per_refresh
                    )
                )
                metrics = self.trainer.update(samples)
                self.pool.sync_replicas()
                self.update_index += 1
                engine.report_nn_metric(
                    self.update_index,
                    int(engine.node_count),
                    metrics.policy_loss,
                    metrics.value_loss,
                    metrics.total_loss,
                    metrics.learning_rate,
                )
                print(
                    f"[nn] update={self.update_index} nodes={int(engine.node_count)} "
                    f"loss={metrics.total_loss:.6g} "
                    f"policy={metrics.policy_loss:.6g} value={metrics.value_loss:.6g}",
                    flush=True,
                )
                _refresh_all_priors(engine, self.pool, self.runtime)
                self._last_status_nodes = int(engine.node_count)
                self._last_status_time = time.monotonic()

        self._status_line(engine, time.monotonic(), force=True)
        self.completed_nodes += int(engine.node_count)
