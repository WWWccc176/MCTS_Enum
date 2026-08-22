#pragma once

#include "mcts/basis.hpp"

#include <vector>

namespace mcts_enum {

class GsoData {
public:
    void recompute(const Basis& basis);

    int dimension() const noexcept { return n_; }
    long square_scale_exp2() const noexcept { return square_scale_exp2_; }
    long double mu(int i, int j) const noexcept {
        return mu_[static_cast<std::size_t>(i) * n_ + j];
    }
    long double g(int i) const noexcept { return g_[static_cast<std::size_t>(i)]; }
    long double log_gh() const noexcept { return log_gh_; }
    double log_potential() const noexcept { return log_potential_; }

private:
    int n_ = 0;
    long square_scale_exp2_ = 0;
    std::vector<long double> mu_;
    std::vector<long double> g_;
    long double log_gh_ = 0.0L;
    double log_potential_ = 0.0;
};

}  // namespace mcts_enum
