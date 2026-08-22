#pragma once

#include "mcts/basis.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace mcts_enum {

struct BasisRefreshResult {
    Basis basis;
    bool completed = false;
    bool changed = false;
    bool accepted = false;
    double potential_before = 0.0;
    double potential_after = 0.0;
    std::string error;
};

BasisRefreshResult lll_refresh(
    const Basis& current,
    const std::vector<std::int64_t>& candidate_coefficients,
    double lll_delta,
    double potential_rel_tolerance);

}  // namespace mcts_enum
