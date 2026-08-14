#pragma once

#include "mcts/gso.hpp"
#include "mcts/types.hpp"

#include <cstdint>
#include <vector>

namespace mcts_enum {

struct CandidateInfo {
    std::int64_t action = 0;
    float normalized_offset = 0.0f;
    float normalized_abs_offset = 0.0f;
    float se_rank = 0.0f;
};

class EnumerationGeometry {
public:
    EnumerationGeometry(const GsoData& gso, const SearchConfig& config)
        : gso_(gso), config_(config) {}

    long double center(int k, const std::vector<std::int64_t>& z) const;
    GeometryInfo legal_interval(
        int k,
        long double rho,
        long double radius_sq_scaled,
        const std::vector<std::int64_t>& z) const;

    std::vector<CandidateInfo> candidates(const GeometryInfo& info) const;
    long double next_rho(int k, long double rho, long double center,
                         std::int64_t action) const;

private:
    const GsoData& gso_;
    const SearchConfig& config_;
};

}  // namespace mcts_enum
