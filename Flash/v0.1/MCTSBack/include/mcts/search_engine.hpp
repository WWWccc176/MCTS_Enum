#pragma once

#include "mcts/basis.hpp"
#include "mcts/enumeration_geometry.hpp"
#include "mcts/gso.hpp"
#include "mcts/result_recorder.hpp"
#include "mcts/tree.hpp"
#include "mcts/types.hpp"

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mcts_enum {

struct EvalRequest {
    std::uint64_t request_id = 0;
    std::int64_t node_id = -1;
    std::vector<float> global_features;
    std::vector<float> recent_residuals;
    std::vector<float> candidate_features;
    std::size_t candidate_count = 0;
};

struct TrainingSample {
    std::int64_t node_id = -1;
    std::vector<float> global_features;
    std::vector<float> recent_residuals;
    std::vector<float> candidate_features;
    std::vector<float> policy_target;
    float value_target = 0.0f;
    bool has_value_target = false;
};

struct StatusSnapshot {
    std::uint32_t phase = 0;
    std::uint64_t nodes = 0;
    std::uint64_t node_budget = 0;
    bool unlimited_nodes = false;
    std::uint64_t best_found_at_node_count = 0;
    std::uint64_t refresh_candidate_found_at_node_count = 0;
    std::uint64_t nodes_since_refresh = 0;
    std::uint64_t root_visits = 0;
    std::uint64_t progress_epoch = 0;
    std::size_t pending = 0;
    std::uint32_t radius_drops_since_global_update = 0;
    double initial_quality_ratio = 0.0;
    double best_quality_ratio = 0.0;
    double refresh_candidate_quality_ratio = 0.0;
    bool refresh_candidate_available = false;
    double best_score = 0.0;
    bool budget_reached = false;
    bool root_closed = false;
};

class SearchEngine {
public:
    SearchEngine(Basis basis, SearchConfig config, bool nn_enabled);

    static SearchEngine from_packet(const std::string& packet,
                                    const SearchConfig& config,
                                    bool nn_enabled);

    double input_quality_ratio() const noexcept { return input_quality_ratio_; }
    std::uint64_t node_count() const noexcept { return tree_.size(); }
    std::uint64_t best_found_at_node_count() const noexcept { return best_found_at_node_count_; }
    std::uint64_t progress_epoch() const noexcept { return progress_epoch_; }
    std::size_t pending_count() const;
    std::uint64_t nodes_since_refresh() const noexcept {
        return static_cast<std::uint64_t>(tree_.size()) - refresh_node_baseline_;
    }
    bool finished() const noexcept;
    std::string diagnostic_status() const;
    StatusSnapshot status_snapshot() const;
    int dimension() const noexcept { return basis_.dimension(); }

    void run_flash();
    std::vector<EvalRequest> collect_inference_batch(std::size_t max_batch);
    void submit_inference(const std::vector<std::uint64_t>& request_ids,
                          const std::vector<std::vector<float>>& logits,
                          const std::vector<float>& values);

    std::vector<TrainingSample> training_samples(std::size_t max_samples) const;
    std::vector<EvalRequest> collect_refresh_batch(std::size_t cursor,
                                                   std::size_t max_batch,
                                                   std::size_t* next_cursor) const;
    void apply_refresh(const std::vector<std::int64_t>& node_ids,
                       const std::vector<std::vector<float>>& logits,
                       const std::vector<float>& values);
    void mark_refresh_complete() noexcept {
        refresh_node_baseline_ = static_cast<std::uint64_t>(tree_.size());
    }

    bool refresh_basis_with_best();

    std::vector<std::int64_t> best_coefficients() const;
    std::vector<std::int64_t> refresh_candidate_coefficients() const;
    std::vector<mpz_class> best_vector() const;
    std::string current_basis_text() const { return basis_.to_text(); }
    std::string current_basis_packet() const { return basis_.to_packet(); }
    double best_score() const noexcept { return best_score_; }
    std::vector<float> gso_features() const;

    void report_nn_metric(const NnMetricRecord& metric) {
        recorder_.add_nn_metric(metric);
    }
    void write_results(const std::string& result_root,
                       const std::string& version,
                       const std::string& run_id);

private:
    struct PendingEvaluation {
        std::uint64_t request_id = 0;
        std::int64_t node_id = -1;
        std::vector<std::pair<std::int64_t, std::size_t>> path;
        bool valid = true;
    };

    bool node_budget_reached() const noexcept;
    bool can_create_node() const noexcept;
    void initialize_episode();
    void initialize_node_geometry(Node& node);
    EvalRequest make_request(const Node& node, std::uint64_t request_id) const;
    std::vector<std::int64_t> reconstruct_coefficients(std::int64_t node_id) const;
    std::vector<float> recent_residual_features(std::int64_t node_id) const;
    std::vector<float> global_features(const Node& node) const;
    std::vector<std::pair<std::int64_t, std::size_t>> select_path_to_leaf(
        std::int64_t* leaf_id, bool* created_node, bool defer_geometry = false);
    void apply_node_geometry(Node& node,
                             GeometryInfo geometry,
                             std::vector<CandidateInfo> candidates);
    std::size_t choose_edge(const Node& node) const;
    std::size_t next_unexpanded_edge(const Node& node) const;
    std::size_t progressive_widening_limit(const Node& node) const;
    void set_policy_and_value(Node& node, const std::vector<float>& logits,
                              float value, bool count_progress = true);
    void initialize_flash_policy(Node& node);
    void backup(const std::vector<std::pair<std::int64_t, std::size_t>>& path,
                double value, bool exact_terminal);
    void tighten_tree_after_radius_update();
    void on_radius_decreased();
    void evaluate_terminal(
        std::int64_t node_id,
        const std::vector<std::pair<std::int64_t, std::size_t>>& path);
    void commit_terminal(
        std::int64_t node_id,
        const std::vector<std::pair<std::int64_t, std::size_t>>& path,
        const std::vector<std::int64_t>& z,
        const mpz_class& norm_sq);
    double terminal_score_from_norm(const mpz_class& norm_sq) const;
    long double radius_sq_scaled() const;
    PhaseSnapshot snapshot_phase(bool basis_refresh_after,
                                 bool basis_refresh_changed,
                                 const std::string& refreshed_basis) const;
    std::vector<PathRecord> best_path_records() const;

    Basis basis_;
    SearchConfig config_;
    bool nn_enabled_ = false;
    GsoData gso_;
    std::vector<float> gso_features_cache_;
    EnumerationGeometry geometry_;
    Tree tree_;
    ResultRecorder recorder_;

    mpz_class r0_sq_ = 0;
    mpz_class best_sq_ = 0;
    std::vector<std::int64_t> best_z_;
    mpz_class refresh_candidate_sq_ = 0;
    std::vector<std::int64_t> refresh_candidate_z_;
    std::uint64_t refresh_candidate_found_at_node_count_ = 0;
    std::int64_t best_terminal_node_id_ = -1;
    std::uint64_t best_found_at_node_count_ = 0;
    std::vector<PathRecord> best_path_snapshot_;
    double best_score_ = 0.0;
    double input_quality_ratio_ = 0.0;

    std::uint64_t next_request_id_ = 1;
    std::unordered_map<std::uint64_t, PendingEvaluation> pending_;
    std::uint64_t progress_epoch_ = 0;
    std::uint64_t radius_epoch_ = 0;
    std::uint64_t refresh_node_baseline_ = 0;
    std::uint32_t radius_drops_since_global_update_ = 0;
    std::uint32_t phase_index_ = 0;
    std::uint64_t completed_phase_nodes_ = 0;
    std::string phase_initial_basis_;
    mutable std::mutex mutex_;
};

}  // namespace mcts_enum
