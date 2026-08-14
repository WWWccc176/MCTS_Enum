#include "mcts/gso.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace mcts_enum {

void GsoData::recompute(const Basis& basis) {
    n_ = basis.dimension();
    cols_ = basis.columns();
    if (n_ <= 0 || cols_ != n_) throw std::runtime_error("GSO requires a square basis");

    scale_exp2_ = 0;
    for (int i = 0; i < n_; ++i) {
        for (int j = 0; j < cols_; ++j) {
            mpz_srcptr value = basis.matrix()[i][j].get_data();
            if (mpz_sgn(value) == 0) continue;
            long exponent = 0;
            (void)mpz_get_d_2exp(&exponent, value);
            scale_exp2_ = std::max(scale_exp2_, exponent);
        }
    }

    std::vector<long double> raw(static_cast<std::size_t>(n_) * cols_, 0.0L);
    for (int i = 0; i < n_; ++i) {
        for (int j = 0; j < cols_; ++j) {
            long exponent = 0;
            const double mantissa = mpz_get_d_2exp(&exponent, basis.matrix()[i][j].get_data());
            raw[static_cast<std::size_t>(i) * cols_ + j] =
                std::ldexp(static_cast<long double>(mantissa), exponent - scale_exp2_);
        }
    }

    mu_.assign(static_cast<std::size_t>(n_) * n_, 0.0L);
    g_.assign(n_, 0.0L);
    log_g_.assign(n_, 0.0L);
    std::vector<long double> orth(static_cast<std::size_t>(n_) * cols_, 0.0L);
    std::vector<long double> current(cols_, 0.0L);

    for (int i = 0; i < n_; ++i) {
        const long double* source = &raw[static_cast<std::size_t>(i) * cols_];
        std::copy(source, source + cols_, current.begin());

        for (int pass = 0; pass < 2; ++pass) {
            for (int j = 0; j < i; ++j) {
                if (!(g_[j] > std::numeric_limits<long double>::min())) {
                    throw std::runtime_error("numerically singular basis during GSO");
                }
                const long double* q = &orth[static_cast<std::size_t>(j) * cols_];
                long double dot = 0.0L;
                for (int c = 0; c < cols_; ++c) dot += current[c] * q[c];
                const long double correction = dot / g_[j];
                mu_[static_cast<std::size_t>(i) * n_ + j] += correction;
                for (int c = 0; c < cols_; ++c) current[c] -= correction * q[c];
            }
        }

        long double squared = 0.0L;
        for (int c = 0; c < cols_; ++c) {
            orth[static_cast<std::size_t>(i) * cols_ + c] = current[c];
            squared += current[c] * current[c];
        }
        if (!(squared > 0.0L) || !std::isfinite(squared)) {
            throw std::runtime_error("non-positive or non-finite GSO norm");
        }
        g_[i] = squared;
        log_g_[i] = std::log(squared) +
                    2.0L * static_cast<long double>(scale_exp2_) * std::log(2.0L);
    }

    const mpz_class exact_det = basis.exact_abs_determinant();
    if (exact_det == 0) {
        throw std::runtime_error("singular basis: exact determinant is zero");
    }
    const long double log_det = Basis::log_positive_mpz(exact_det);
    const long double pi = std::acos(-1.0L);
    log_gh_ =
        (std::lgamma(static_cast<long double>(n_) / 2.0L + 1.0L) -
         static_cast<long double>(n_) / 2.0L * std::log(pi) + log_det) /
        static_cast<long double>(n_);
    gh_scaled_ = std::exp(
        log_gh_ - static_cast<long double>(scale_exp2_) * std::log(2.0L));
}

std::vector<float> GsoData::normalized_features() const {
    std::vector<float> out(static_cast<std::size_t>(n_) * 2, 0.0f);
    if (n_ == 0) return out;
    long double mean = 0.0L;
    for (const auto x : log_g_) mean += x;
    mean /= static_cast<long double>(n_);
    long double variance = 0.0L;
    for (const auto x : log_g_) {
        const long double d = x - mean;
        variance += d * d;
    }
    variance /= static_cast<long double>(n_);
    const long double stddev = std::sqrt(std::max(variance, 1.0e-30L));
    for (int i = 0; i < n_; ++i) {
        out[static_cast<std::size_t>(i) * 2] =
            static_cast<float>((log_g_[i] - mean) / stddev);
        out[static_cast<std::size_t>(i) * 2 + 1] =
            n_ > 1 ? static_cast<float>(i) / static_cast<float>(n_ - 1) : 0.0f;
    }
    return out;
}

}  // namespace mcts_enum
