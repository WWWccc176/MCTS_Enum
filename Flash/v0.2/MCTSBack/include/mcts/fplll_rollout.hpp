#pragma once

#include "mcts/basis.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace mcts_enum {

struct RolloutResult {
    std::uint64_t enumeration_nodes = 0;
    std::uint64_t solution_callbacks = 0;
    std::vector<std::uint64_t> nodes_per_level;
    std::vector<std::vector<std::int64_t>> stored_coefficients;
};

class FplllRolloutWorker {
public:
    explicit FplllRolloutWorker(const Basis& basis);
    ~FplllRolloutWorker();
    FplllRolloutWorker(FplllRolloutWorker&&) noexcept;
    FplllRolloutWorker& operator=(FplllRolloutWorker&&) noexcept;
    FplllRolloutWorker(const FplllRolloutWorker&) = delete;
    FplllRolloutWorker& operator=(const FplllRolloutWorker&) = delete;

    RolloutResult enumerate_subtree(
        const std::vector<std::int64_t>& prefix_high_to_low,
        const mpz_class& radius_squared,
        std::size_t stored_solutions);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace mcts_enum
