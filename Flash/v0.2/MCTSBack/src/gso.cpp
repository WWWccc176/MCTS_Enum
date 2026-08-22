#include "mcts/gso.hpp"

#include <fplll.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace mcts_enum {

void GsoData::recompute(const Basis& basis) {
    n_ = basis.dimension();
    if (n_ <= 0 || basis.columns() != n_) {
        throw std::runtime_error("GSO requires a square basis");
    }

    Matrix work = basis.matrix();
    Matrix u;
    Matrix u_inv_t;
    fplll::MatGSO<fplll::Z_NR<mpz_t>, fplll::FP_NR<long double>> m(
        work, u, u_inv_t, fplll::GSO_ROW_EXPO);
    if (!m.update_gso()) {
        throw std::runtime_error("fplll MatGSO update failed");
    }

    mu_.assign(static_cast<std::size_t>(n_) * n_, 0.0L);
    std::vector<long double> log_g(static_cast<std::size_t>(n_), 0.0L);
    const long double log2 = std::log(2.0L);
    long double max_log_g = -std::numeric_limits<long double>::infinity();

    for (int i = 0; i < n_; ++i) {
        long expo = 0;
        const auto& r = m.get_r_exp(i, i, expo);
        const long double rv = r.get_data();
        if (!(rv > 0.0L) || !std::isfinite(rv)) {
            throw std::runtime_error("non-positive/non-finite fplll GSO diagonal");
        }
        log_g[static_cast<std::size_t>(i)] = std::log(rv) + static_cast<long double>(expo) * log2;
        max_log_g = std::max(max_log_g, log_g[static_cast<std::size_t>(i)]);

        for (int j = 0; j < i; ++j) {
            fplll::FP_NR<long double> v;
            m.get_mu(v, i, j);
            const long double x = v.get_data();
            if (!std::isfinite(x)) throw std::runtime_error("non-finite fplll GSO mu");
            mu_[static_cast<std::size_t>(i) * n_ + j] = x;
        }
    }

    square_scale_exp2_ = static_cast<long>(std::floor(max_log_g / log2));
    g_.resize(static_cast<std::size_t>(n_));
    for (int i = 0; i < n_; ++i) {
        const long double scaled = std::exp(
            log_g[static_cast<std::size_t>(i)] -
            static_cast<long double>(square_scale_exp2_) * log2);
        if (!(scaled > 0.0L) || !std::isfinite(scaled)) {
            throw std::runtime_error("scaled GSO diagonal is invalid");
        }
        g_[static_cast<std::size_t>(i)] = scaled;
    }

    const mpz_class det = basis.exact_abs_determinant();
    if (det <= 0) throw std::runtime_error("singular basis");
    const long double log_det = Basis::log_positive_mpz(det);
    const long double pi = std::acos(-1.0L);
    log_gh_ = (
        std::lgamma(static_cast<long double>(n_) / 2.0L + 1.0L) -
        static_cast<long double>(n_) / 2.0L * std::log(pi) + log_det
    ) / static_cast<long double>(n_);

    long double potential = 0.0L;
    for (int i = 0; i < n_; ++i) {
        potential += 0.5L * static_cast<long double>(n_ - i) *
                     log_g[static_cast<std::size_t>(i)];
    }
    log_potential_ = static_cast<double>(potential);
}

}  // namespace mcts_enum
