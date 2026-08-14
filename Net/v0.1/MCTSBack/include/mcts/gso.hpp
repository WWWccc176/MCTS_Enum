#pragma once

#include "mcts/basis.hpp"

#include <cstdint>
#include <vector>

namespace mcts_enum {

class GsoData {
public:
    void recompute(const Basis& basis);

    int dimension() const noexcept { return n_; }
    long scale_exp2() const noexcept { return scale_exp2_; }
    long double mu(int i, int j) const noexcept {
        return mu_[static_cast<std::size_t>(i) * n_ + j];
    }
    long double g(int i) const noexcept { return g_[i]; }
    const std::vector<long double>& g_values() const noexcept { return g_; }
    long double log_gh() const noexcept { return log_gh_; }
    long double gh_scaled() const noexcept { return gh_scaled_; }
    std::vector<float> normalized_features() const;

private:
    int n_ = 0;
    int cols_ = 0;
    long scale_exp2_ = 0;
    std::vector<long double> mu_;
    std::vector<long double> g_;
    std::vector<long double> log_g_;
    long double log_gh_ = 0.0L;
    long double gh_scaled_ = 0.0L;
};

}  // namespace mcts_enum
