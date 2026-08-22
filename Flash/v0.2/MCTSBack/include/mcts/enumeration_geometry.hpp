#pragma once

#include "mcts/gso.hpp"
#include "mcts/types.hpp"

#include <cstdint>
#include <vector>

namespace mcts_enum {

class EnumerationGeometry {
public:
    EnumerationGeometry(const GsoData& gso, const SearchConfig& config)
        : gso_(gso), config_(config) {}

    long double center_from_prefix(
        int k, const std::vector<std::int64_t>& prefix_high_to_low) const;

    GeometryInfo legal_interval(
        int k,
        long double rho,
        long double radius_sq_scaled,
        const std::vector<std::int64_t>& prefix_high_to_low) const;

    GeometryInfo legal_interval_from_center(
        int k,
        long double rho,
        long double radius_sq_scaled,
        long double center) const;

    long double next_rho(
        int k, long double rho, long double center, std::int64_t action) const;

private:
    const GsoData& gso_;
    const SearchConfig& config_;
};

}  // namespace mcts_enum
