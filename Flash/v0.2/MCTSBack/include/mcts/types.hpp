#pragma once

#include <gmpxx.h>

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace mcts_enum {

inline constexpr const char* kBackendVersion = "v0.2";

struct SearchConfig {
    std::uint64_t node_budget = 0;
    bool unlimited_nodes = false;
    std::uint32_t search_threads = 1;
    std::uint32_t rollout_dimensions = 10;
    std::uint32_t rollout_solutions = 8;
    double w_m = 0.25;
    double lambda_puct = 1.5;
    double cpw = 2.0;
    double dpw = 0.5;
    double numeric_guard_rel = 1.0e-12;
    double numeric_guard_abs = 1.0e-18;
    double lll_delta = 0.999;
    double refresh_potential_rel_tolerance = 1.0e-10;
};

struct GeometryInfo {
    int k = -1;
    long double center = 0.0L;
    long double delta = 0.0L;
    long double rho = 0.0L;
    long double radius_sq = 0.0L;
    std::int64_t lo = 0;
    std::int64_t hi = -1;
    bool pruned = false;
};

struct EdgeStats {
    std::int64_t action = 0;
    double prior_weight = 1.0;
    std::uint64_t visits = 0;
    std::uint32_t in_flight = 0;
    double w = 0.0;
    double q = 0.0;
    double m = -std::numeric_limits<double>::infinity();
    bool has_exact_m = false;
    std::int64_t child_id = -1;
    bool expanded = false;
    bool active = true;
};

struct PathRecord {
    std::uint64_t depth = 0;
    int coefficient_index = -1;
    std::int64_t node_id = -1;
    std::int64_t parent_id = -1;
    std::int64_t action = 0;
    std::uint64_t nodes_at_depth = 0;
    long double parent_rho = 0.0L;
    long double rho = 0.0L;
    long double radius_sq = 0.0L;
    long double center = 0.0L;
    long double delta = 0.0L;
    std::int64_t legal_lo = 0;
    std::int64_t legal_hi = -1;
    std::uint64_t edge_visits = 0;
    double q = 0.0;
    double m = 0.0;
};

struct StatusSnapshot {
    std::uint32_t phase = 0;
    std::uint64_t tree_nodes = 0;
    std::uint64_t enumeration_nodes = 0;
    std::uint64_t work_nodes = 0;
    std::uint64_t node_budget = 0;
    bool unlimited_nodes = false;
    std::uint64_t rollout_jobs = 0;
    std::uint64_t exact_candidates = 0;
    std::uint64_t best_found_at_work_node_count = 0;
    std::uint64_t refresh_candidate_found_at_work_node_count = 0;
    std::uint64_t root_visits = 0;
    std::uint32_t radius_drops = 0;
    double initial_quality_ratio = 0.0;
    double best_quality_ratio = 0.0;
    double search_radius_quality_ratio = 0.0;
    double refresh_candidate_quality_ratio = 0.0;
    bool refresh_candidate_available = false;
    bool budget_reached = false;
    bool root_closed = false;
    std::uint64_t active_rollouts = 0;
};

struct PhaseSnapshot {
    std::uint32_t phase_index = 0;
    std::string initial_basis_text;
    std::string refreshed_basis_text;
    std::uint64_t tree_nodes = 0;
    std::uint64_t enumeration_nodes = 0;
    std::uint64_t work_nodes = 0;
    std::uint64_t rollout_jobs = 0;
    std::uint64_t exact_candidates = 0;
    std::uint64_t best_found_at_work_node_count = 0;
    std::vector<std::uint64_t> nodes_per_depth;
    std::vector<std::uint64_t> enumeration_nodes_per_level;
    mpz_class initial_radius_squared = 0;
    double initial_radius_over_gh = 0.0;
    long double log_gh = 0.0L;
    mpz_class best_squared_norm = 0;
    std::vector<std::int64_t> best_coefficients;
    std::vector<mpz_class> best_vector;
    std::vector<PathRecord> best_path;
    bool budget_reached = false;
    bool tree_exhausted = false;
    bool basis_refresh_after = false;
    bool basis_refresh_changed = false;
    bool basis_refresh_accepted = false;
    double refresh_potential_before = 0.0;
    double refresh_potential_after = 0.0;
};

}  // namespace mcts_enum
