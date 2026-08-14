#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace mcts_enum {

inline constexpr const char* kBackendVersion = "v0.1";
inline constexpr double kRequiredInputQuality = 1.2;
inline constexpr std::uint64_t kDefaultRefreshInterval = 5000;
inline constexpr std::uint32_t kDefaultRadiusGlobalUpdateInterval = 2;
inline constexpr double kDefaultExtremeWeight = 0.25;

struct SearchConfig {
    std::uint64_t node_budget = 0;
    bool unlimited_nodes = false;
    std::uint64_t refresh_interval = kDefaultRefreshInterval;
    std::uint32_t radius_global_update_interval = kDefaultRadiusGlobalUpdateInterval;
    double w_m = kDefaultExtremeWeight;
    double lambda_puct = 1.5;
    double cpw = 2.0;
    double dpw = 0.5;
    double policy_mix = 0.5;
    double visit_temperature = 1.0;
    double quality_gate = kRequiredInputQuality;
    double numeric_guard_rel = 1.0e-12;
    double numeric_guard_abs = 1.0e-18;
    std::uint32_t max_legal_actions = 65536;
    std::uint32_t recent_residual_count = 8;
    std::uint32_t search_threads = 1;
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
    double prior = 0.0;
    double policy_prior = 0.0;
    std::uint64_t visits = 0;
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
    long scale_exp2 = 0;
    long double parent_rho = 0.0L;
    long double rho = 0.0L;
    long double incremental_cost = 0.0L;
    long double radius_sq = 0.0L;
    long double remaining_sq = 0.0L;
    long double center = 0.0L;
    long double delta = 0.0L;
    std::int64_t legal_lo = 0;
    std::int64_t legal_hi = -1;
    double normalized_offset = 0.0;
    double prior = 0.0;
    double policy_prior = 0.0;
    std::uint64_t edge_visits = 0;
    double q = 0.0;
    double m = 0.0;
};

struct NnMetricRecord {
    std::uint64_t update_index = 0;
    std::uint64_t node_count = 0;
    double policy_loss = 0.0;
    double value_loss = 0.0;
    double total_loss = 0.0;
    double learning_rate = 0.0;
    bool enabled = false;
};

}  // namespace mcts_enum
