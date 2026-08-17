#include "mcts/basis_refresh.hpp"

#include <exception>

namespace mcts_enum {
namespace {

bool equal_basis(const Matrix& a, const Matrix& b) {
    if (a.get_rows() != b.get_rows() || a.get_cols() != b.get_cols()) return false;
    for (int i = 0; i < a.get_rows(); ++i) {
        for (int j = 0; j < a.get_cols(); ++j) {
            if (mpz_cmp(a[i][j].get_data(), b[i][j].get_data()) != 0) return false;
        }
    }
    return true;
}

bool is_single_basis_vector(const std::vector<std::int64_t>& z) {
    int nonzero = 0;
    for (const auto value : z) {
        if (value == 0) continue;
        ++nonzero;
        if (value != 1 && value != -1) return false;
        if (nonzero > 1) return false;
    }
    return nonzero == 1;
}

}  // namespace

BasisRefreshResult lll_refresh(
    const Basis& current,
    const std::vector<std::int64_t>& candidate_coefficients,
    double lll_delta) {
    BasisRefreshResult result;
    try {
        const int n = current.dimension();
        const int cols = current.columns();
        if (static_cast<int>(candidate_coefficients.size()) != n) {
            result.error = "basis refresh coefficient dimension mismatch";
            return result;
        }
        if (is_single_basis_vector(candidate_coefficients)) {
            result.error = "basis refresh candidate must not be a single basis vector";
            return result;
        }

        const auto candidate = current.exact_vector(candidate_coefficients);
        Matrix expanded(n + 1, cols);

        // The discovered non-basis vector is deliberately inserted FIRST.
        for (int j = 0; j < cols; ++j) {
            mpz_set(expanded[0][j].get_data(), candidate[j].get_mpz_t());
        }
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < cols; ++j) {
                expanded[i + 1][j] = current.matrix()[i][j];
            }
        }

        // Lightweight refresh: LLL only. BKZ remains HeavyBack-only.
        fplll::lll_reduction(expanded, lll_delta);

        // The inserted lattice vector is dependent on the old basis, so LLL on
        // n+1 rows may emit zero rows. Remove zeros and restore exactly n rows.
        Matrix reduced(n, cols);
        int row = 0;
        for (int i = 0; i < n + 1 && row < n; ++i) {
            bool zero = true;
            for (int j = 0; j < cols; ++j) {
                if (expanded[i][j].sgn() != 0) {
                    zero = false;
                    break;
                }
            }
            if (zero) continue;
            for (int j = 0; j < cols; ++j) reduced[row][j] = expanded[i][j];
            ++row;
        }
        if (row != n) {
            result.error = "LLL refresh did not restore full rank after zero-row removal";
            return result;
        }

        fplll::lll_reduction(reduced, lll_delta);

        result.changed = !equal_basis(current.matrix(), reduced);
        result.basis = Basis(std::move(reduced));
        result.completed = true;
        return result;
    } catch (const std::exception& ex) {
        result.error = ex.what();
        return result;
    } catch (...) {
        result.error = "unknown exception during LLL refresh";
        return result;
    }
}

}  // namespace mcts_enum
