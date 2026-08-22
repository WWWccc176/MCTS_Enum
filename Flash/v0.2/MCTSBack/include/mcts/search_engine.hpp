#pragma once

#include "mcts/basis.hpp"
#include "mcts/basis_refresh.hpp"
#include "mcts/enumeration_geometry.hpp"
#include "mcts/fplll_rollout.hpp"
#include "mcts/gso.hpp"
#include "mcts/result_recorder.hpp"
#include "mcts/tree.hpp"
#include "mcts/types.hpp"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace mcts_enum {

class SearchEngine {
public:
    SearchEngine(Basis basis, SearchConfig config, bool nn_enabled);

    double input_quality_ratio() const noexcept { return input_quality_ratio_; }
    int dimension() const noexcept { return basis_.dimension(); }
    std::uint64_t node_count() const noexcept;
    std::uint64_t work_node_count() const noexcept;
    bool finished() const;
    std::string diagnostic_status() const;
    StatusSnapshot status_snapshot() const;

    void run_flash();
    bool refresh_basis_with_best();

    std::vector<std::int64_t> refresh_candidate_coefficients() const;
    std::vector<mpz_class> best_vector() const;
    std::string current_basis_text() const;
    std::string current_basis_packet() const;

    void write_results(
        const std::string& result_root,
        const std::string& version,
        const std::string& run_id);

private:
    using PathStep = std::pair<std::int64_t, std::size_t>;

    struct RolloutJob {
        std::int64_t leaf_id = -1;
        std::vector<PathStep> path;
        std::vector<std::int64_t> prefix_high_to_low;
        mpz_class radius_squared = 0;
        std::uint64_t radius_epoch = 0;
    };

    struct ExactCandidate {
        std::vector<std::int64_t> z;
        mpz_class norm_sq = 0;
    };

    void initialize_episode();
    std::pair<int, mpz_class> shortest_basis_row() const;
    long double radius_sq_scaled_locked() const;
    bool budget_reached_locked() const noexcept;
    std::uint64_t work_nodes_locked() const noexcept;

    void refresh_geometry_locked(Node& node);
    bool generate_next_se_edge_locked(Node& node);
    void ensure_widened_locked(Node& node);
    std::size_t choose_edge_locked(const Node& node) const;
    bool reserve_rollout_locked(RolloutJob& job);
    bool is_rollout_frontier(const Node& node) const noexcept;

    void reserve_path_locked(const RolloutJob& job);
    void release_path_locked(const RolloutJob& job);
    void backup_locked(
        const RolloutJob& job,
        double value,
        bool has_exact,
        double exact_score);
    void propagate_closed_locked(const RolloutJob& job);
    bool node_can_close_locked(const Node& node) const;

    void commit_rollout_locked(
        const RolloutJob& job,
        const RolloutResult& rollout,
        const std::vector<ExactCandidate>& exact_candidates);

    double rollout_value(
        const mpz_class& snapshot_radius_sq,
        const std::vector<ExactCandidate>& exact_candidates) const;
    double phase_score_from_norm(const mpz_class& norm_sq) const;
    bool is_single_basis_vector(const std::vector<std::int64_t>& z) const;
    bool is_zero_coefficients(const std::vector<std::int64_t>& z) const;

    std::vector<PathRecord> build_path_records_locked(
        const std::vector<std::int64_t>& z, const RolloutJob& job) const;
    std::vector<PathRecord> best_path_records_locked() const;
    PhaseSnapshot snapshot_phase_locked(
        bool refresh_after,
        const BasisRefreshResult* refresh_result) const;

    Basis basis_;
    SearchConfig config_;
    bool nn_enabled_ = false;
    GsoData gso_;
    EnumerationGeometry geometry_;
    Tree tree_;
    ResultRecorder recorder_;

    mpz_class phase_initial_radius_sq_ = 0;
    mpz_class phase_best_sq_ = 0;
    std::vector<std::int64_t> phase_best_z_;
    std::vector<mpz_class> phase_best_vector_;
    std::uint64_t phase_best_found_at_work_ = 0;
    std::vector<PathRecord> phase_best_path_;

    mpz_class overall_best_sq_ = 0;
    std::vector<mpz_class> overall_best_vector_;

    mpz_class refresh_candidate_sq_ = 0;
    std::vector<std::int64_t> refresh_candidate_z_;
    std::uint64_t refresh_candidate_found_at_work_ = 0;

    double input_quality_ratio_ = 0.0;
    std::uint32_t phase_index_ = 0;
    std::uint64_t radius_epoch_ = 0;
    std::uint32_t radius_drops_ = 0;
    bool phase_snapshot_committed_ = false;
    std::string phase_initial_basis_;

    std::atomic<std::uint64_t> enumeration_nodes_{0};
    std::atomic<std::uint64_t> rollout_jobs_{0};
    std::atomic<std::uint64_t> exact_candidates_{0};
    std::atomic<std::uint64_t> active_rollouts_{0};
    std::vector<std::uint64_t> enumeration_nodes_per_level_;

    mutable std::mutex mutex_;
};

}  // namespace mcts_enum
