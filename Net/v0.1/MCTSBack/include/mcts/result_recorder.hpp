#pragma once

#include "mcts/basis.hpp"
#include "mcts/types.hpp"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace mcts_enum {

struct PhaseSnapshot {
    std::uint32_t phase_index = 0;
    std::string initial_basis_text;
    std::uint64_t total_nodes = 0;
    std::uint64_t best_found_at_node_count = 0;
    std::vector<std::uint64_t> nodes_per_depth;
    mpz_class initial_radius_squared = 0;
    double initial_b1_over_gh = 0.0;
    long double log_gh = 0.0L;
    long gso_scale_exp2 = 0;
    std::vector<mpz_class> best_vector;
    std::vector<std::int64_t> best_coefficients;
    mpz_class best_squared_norm = 0;
    double best_score = 0.0;
    std::vector<PathRecord> best_path;
    bool budget_reached = false;
    bool tree_exhausted = false;
    bool basis_refresh_after = false;
    bool basis_refresh_changed = false;
    std::string refreshed_basis_text;
};

class ResultRecorder {
public:
    void set_nn_enabled(bool enabled) noexcept { nn_enabled_ = enabled; }
    void set_search_config(const SearchConfig& config) {
        search_config_ = config;
        has_search_config_ = true;
    }
    void add_phase(PhaseSnapshot snapshot);
    void add_nn_metric(const NnMetricRecord& metric);
    void write(const std::string& result_root,
               const std::string& version,
               const std::string& run_id) const;

private:
    bool nn_enabled_ = false;
    bool has_search_config_ = false;
    SearchConfig search_config_;
    std::vector<PhaseSnapshot> phases_;
    std::vector<NnMetricRecord> nn_metrics_;
};

}  // namespace mcts_enum
