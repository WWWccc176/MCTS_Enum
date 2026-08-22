#include "mcts/fplll_rollout.hpp"

#include <fplll.h>
#include <fplll/enum/enumerate.h>
#include <fplll/enum/evaluator.h>

#include <cmath>
#include <limits>
#include <stdexcept>

namespace mcts_enum {

struct FplllRolloutWorker::Impl {
    Matrix basis;
    Matrix u;
    Matrix u_inv_t;
    fplll::MatGSO<fplll::Z_NR<mpz_t>, fplll::FP_NR<long double>> gso;

    explicit Impl(const Basis& input)
        : basis(input.matrix()),
          gso(basis, u, u_inv_t, fplll::GSO_ROW_EXPO) {
        if (!gso.update_gso()) throw std::runtime_error("fplll rollout GSO update failed");
    }
};

FplllRolloutWorker::FplllRolloutWorker(const Basis& basis)
    : impl_(std::make_unique<Impl>(basis)) {}

FplllRolloutWorker::~FplllRolloutWorker() = default;
FplllRolloutWorker::FplllRolloutWorker(FplllRolloutWorker&&) noexcept = default;
FplllRolloutWorker& FplllRolloutWorker::operator=(FplllRolloutWorker&&) noexcept = default;

RolloutResult FplllRolloutWorker::enumerate_subtree(
    const std::vector<std::int64_t>& prefix_high_to_low,
    const mpz_class& radius_squared,
    std::size_t stored_solutions) {
    if (radius_squared <= 0) throw std::runtime_error("rollout radius must be positive");
    if (stored_solutions == 0) stored_solutions = 1;

    std::vector<fplll::enumxt> subtree;
    subtree.reserve(prefix_high_to_low.size());
    for (auto it = prefix_high_to_low.rbegin(); it != prefix_high_to_low.rend(); ++it) {
        const long double value = static_cast<long double>(*it);
        if (std::fabs(value) > static_cast<long double>(1ULL << 53)) {
            throw std::runtime_error("prefix coefficient exceeds exact fplll enumxt integer range");
        }
        subtree.push_back(static_cast<fplll::enumxt>(*it));
    }

    long radius_exponent = 0;
    const double radius_mantissa = mpz_get_d_2exp(&radius_exponent, radius_squared.get_mpz_t());
    if (!(radius_mantissa > 0.0) || !std::isfinite(radius_mantissa)) {
        throw std::runtime_error("failed to normalize exact rollout radius");
    }

    fplll::FP_NR<long double> max_dist;
    max_dist = static_cast<long double>(radius_mantissa);

    fplll::FastEvaluator<fplll::FP_NR<long double>> evaluator(
        stored_solutions,
        fplll::EVALSTRATEGY_BEST_N_SOLUTIONS,
        false);
    fplll::Enumeration<fplll::Z_NR<mpz_t>, fplll::FP_NR<long double>> enumeration(
        impl_->gso,
        evaluator);

    const int dimension = impl_->basis.get_rows();
    enumeration.enumerate(
        0,
        dimension,
        max_dist,
        radius_exponent,
        std::vector<fplll::FP_NR<long double>>(),
        subtree,
        std::vector<fplll::enumf>(),
        false,
        false);

    RolloutResult out;
    out.enumeration_nodes = enumeration.get_nodes();
    out.solution_callbacks = static_cast<std::uint64_t>(evaluator.sol_count);
    out.nodes_per_level.assign(static_cast<std::size_t>(dimension), 0);
    const auto node_array = enumeration.get_nodes_array();
    const int remaining_last = dimension - static_cast<int>(prefix_high_to_low.size()) - 1;
    for (int k = 0; k <= remaining_last && k < dimension; ++k) {
        out.nodes_per_level[static_cast<std::size_t>(k)] = node_array[static_cast<std::size_t>(k)];
    }
    out.stored_coefficients.reserve(evaluator.size());

    for (auto it = evaluator.begin(); it != evaluator.end(); ++it) {
        const auto& coords = it->second;
        if (static_cast<int>(coords.size()) != dimension) {
            throw std::runtime_error("fplll rollout returned wrong coefficient dimension");
        }
        std::vector<std::int64_t> z(static_cast<std::size_t>(dimension), 0);
        for (int i = 0; i < dimension; ++i) {
            const long double value = coords[static_cast<std::size_t>(i)].get_data();
            if (!std::isfinite(value) ||
                value < static_cast<long double>(std::numeric_limits<std::int64_t>::min()) ||
                value > static_cast<long double>(std::numeric_limits<std::int64_t>::max())) {
                throw std::runtime_error("fplll rollout coefficient is outside int64 range");
            }
            const auto rounded = static_cast<std::int64_t>(std::llround(value));
            if (std::fabs(value - static_cast<long double>(rounded)) > 0.25L) {
                throw std::runtime_error("fplll rollout returned a non-integral coefficient");
            }
            z[static_cast<std::size_t>(i)] = rounded;
        }
        out.stored_coefficients.push_back(std::move(z));
    }
    return out;
}

}  // namespace mcts_enum
