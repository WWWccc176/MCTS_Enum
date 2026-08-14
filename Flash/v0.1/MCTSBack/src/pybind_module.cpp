#include "mcts/search_engine.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace mcts_enum;

namespace {

py::dict request_to_dict(const EvalRequest& request) {
    py::dict out;
    out["request_id"] = request.request_id;
    out["node_id"] = request.node_id;
    out["global"] = request.global_features;
    out["recent"] = request.recent_residuals;
    out["candidates"] = request.candidate_features;
    out["candidate_count"] = request.candidate_count;
    return out;
}

py::dict sample_to_dict(const TrainingSample& sample) {
    py::dict out;
    out["node_id"] = sample.node_id;
    out["global"] = sample.global_features;
    out["recent"] = sample.recent_residuals;
    out["candidates"] = sample.candidate_features;
    out["policy_target"] = sample.policy_target;
    out["value_target"] = sample.value_target;
    out["has_value_target"] = sample.has_value_target;
    return out;
}

py::list requests_to_list(const std::vector<EvalRequest>& requests) {
    py::list out;
    for (const auto& request : requests) out.append(request_to_dict(request));
    return out;
}

}  // namespace

PYBIND11_MODULE(mcts_enum_backend, module) {
    module.doc() =
        "Lightweight MCTS-in-enumeration backend: exact MPZ terminal evaluation, "
        "numerically guarded GSO bounds, PUCT, progressive widening, and in-memory NN batching";

    py::class_<SearchConfig>(module, "SearchConfig")
        .def(py::init<>())
        .def_readwrite("node_budget", &SearchConfig::node_budget)
        .def_readwrite("unlimited_nodes", &SearchConfig::unlimited_nodes)
        .def_readwrite("refresh_interval", &SearchConfig::refresh_interval)
        .def_readwrite("radius_global_update_interval", &SearchConfig::radius_global_update_interval)
        .def_readwrite("w_m", &SearchConfig::w_m)
        .def_readwrite("lambda_puct", &SearchConfig::lambda_puct)
        .def_readwrite("cpw", &SearchConfig::cpw)
        .def_readwrite("dpw", &SearchConfig::dpw)
        .def_readwrite("policy_mix", &SearchConfig::policy_mix)
        .def_readwrite("visit_temperature", &SearchConfig::visit_temperature)
        .def_readwrite("quality_gate", &SearchConfig::quality_gate)
        .def_readwrite("numeric_guard_rel", &SearchConfig::numeric_guard_rel)
        .def_readwrite("numeric_guard_abs", &SearchConfig::numeric_guard_abs)
        .def_readwrite("max_legal_actions", &SearchConfig::max_legal_actions)
        .def_readwrite("recent_residual_count", &SearchConfig::recent_residual_count)
        .def_readwrite("search_threads", &SearchConfig::search_threads);

    py::class_<SearchEngine>(module, "SearchEngine")
        .def(py::init([](py::bytes packet, const SearchConfig& config, bool nn_enabled) {
            const std::string bytes = packet;
            return std::make_unique<SearchEngine>(Basis::from_packet(bytes), config, nn_enabled);
        }), py::arg("basis_packet"), py::arg("config"), py::arg("nn_enabled"))
        .def_property_readonly("dimension", &SearchEngine::dimension)
        .def_property_readonly("node_count", &SearchEngine::node_count)
        .def_property_readonly("best_found_at_node_count", &SearchEngine::best_found_at_node_count)
        .def_property_readonly("progress_epoch", &SearchEngine::progress_epoch)
        .def_property_readonly("pending_count", &SearchEngine::pending_count)
        .def_property_readonly("nodes_since_refresh", &SearchEngine::nodes_since_refresh)
        .def_property_readonly("input_quality_ratio", &SearchEngine::input_quality_ratio)
        .def_property_readonly("best_score", &SearchEngine::best_score)
        .def("gso_features", &SearchEngine::gso_features)
        .def_property_readonly("finished", &SearchEngine::finished)
        .def("diagnostic_status", &SearchEngine::diagnostic_status)
        .def("status", [](const SearchEngine& engine) {
            const auto s = engine.status_snapshot();
            py::dict out;
            out["phase"] = s.phase;
            out["nodes"] = s.nodes;
            if (s.unlimited_nodes) out["node_budget"] = -1;
            else out["node_budget"] = s.node_budget;
            out["best_found_at_node_count"] = s.best_found_at_node_count;
            out["refresh_candidate_found_at_node_count"] = s.refresh_candidate_found_at_node_count;
            out["nodes_since_refresh"] = s.nodes_since_refresh;
            out["root_visits"] = s.root_visits;
            out["progress_epoch"] = s.progress_epoch;
            out["pending"] = s.pending;
            out["radius_drops_since_global_update"] = s.radius_drops_since_global_update;
            out["initial_quality_ratio"] = s.initial_quality_ratio;
            out["best_quality_ratio"] = s.best_quality_ratio;
            out["refresh_candidate_quality_ratio"] = s.refresh_candidate_quality_ratio;
            out["refresh_candidate_available"] = s.refresh_candidate_available;
            out["best_score"] = s.best_score;
            out["budget_reached"] = s.budget_reached;
            out["root_closed"] = s.root_closed;
            return out;
        })
        .def("run_flash", &SearchEngine::run_flash, py::call_guard<py::gil_scoped_release>())
        .def("collect_inference_batch", [](SearchEngine& engine, std::size_t batch_size) {
            return requests_to_list(engine.collect_inference_batch(batch_size));
        }, py::arg("batch_size"))
        .def("submit_inference", &SearchEngine::submit_inference,
             py::arg("request_ids"), py::arg("logits"), py::arg("values"))
        .def("training_samples", [](const SearchEngine& engine, std::size_t max_samples) {
            py::list out;
            for (const auto& sample : engine.training_samples(max_samples)) {
                out.append(sample_to_dict(sample));
            }
            return out;
        }, py::arg("max_samples"))
        .def("collect_refresh_batch", [](const SearchEngine& engine,
                                          std::size_t cursor,
                                          std::size_t batch_size) {
            std::size_t next_cursor = cursor;
            const auto requests = engine.collect_refresh_batch(cursor, batch_size, &next_cursor);
            py::dict out;
            out["requests"] = requests_to_list(requests);
            out["next_cursor"] = next_cursor;
            return out;
        }, py::arg("cursor"), py::arg("batch_size"))
        .def("apply_refresh", &SearchEngine::apply_refresh,
             py::arg("node_ids"), py::arg("logits"), py::arg("values"))
        .def("mark_refresh_complete", &SearchEngine::mark_refresh_complete)
        .def("refresh_basis_with_best", &SearchEngine::refresh_basis_with_best,
             py::call_guard<py::gil_scoped_release>())
        .def("best_coefficients", &SearchEngine::best_coefficients)
        .def("refresh_candidate_coefficients", &SearchEngine::refresh_candidate_coefficients)
        .def("best_vector", [](const SearchEngine& engine) {
            py::list out;
            for (const auto& value : engine.best_vector()) out.append(value.get_str());
            return out;
        })
        .def("current_basis_text", &SearchEngine::current_basis_text)
        .def("current_basis_packet", [](const SearchEngine& engine) {
            return py::bytes(engine.current_basis_packet());
        })
        .def("report_nn_metric", [](SearchEngine& engine,
                                    std::uint64_t update_index,
                                    std::uint64_t node_count,
                                    double policy_loss,
                                    double value_loss,
                                    double total_loss,
                                    double learning_rate) {
            NnMetricRecord metric;
            metric.enabled = true;
            metric.update_index = update_index;
            metric.node_count = node_count;
            metric.policy_loss = policy_loss;
            metric.value_loss = value_loss;
            metric.total_loss = total_loss;
            metric.learning_rate = learning_rate;
            engine.report_nn_metric(metric);
        })
        .def("write_results", &SearchEngine::write_results,
             py::arg("result_root"), py::arg("version"), py::arg("run_id"));

    module.def("encode_basis_text", [](const std::string& text) {
        return py::bytes(Basis::from_text(text).to_packet());
    }, py::arg("basis_text"));
    module.attr("BACKEND_VERSION") = "v0.1";
    module.attr("REQUIRED_INPUT_QUALITY") = kRequiredInputQuality;
}
