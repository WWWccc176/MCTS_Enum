#pragma once

#include "mcts/basis.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace mcts_enum {

struct BasisRefreshResult {
    bool completed = false;
    bool changed = false;
    std::string error;
    Basis basis;
};

// Insert one discovered non-basis lattice vector at row 0, run LLL only,
// discard zero rows created by the linear dependency, and restore rank n.
BasisRefreshResult lll_refresh(
    const Basis& current,
    const std::vector<std::int64_t>& candidate_coefficients,
    double lll_delta = 0.999);

}  // namespace mcts_enum
