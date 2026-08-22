#include "mcts/basis_refresh.hpp"

#include "mcts/gso.hpp"

#include <cmath>
#include <exception>
#include <stdexcept>

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

bool is_zero_vector(const std::vector<mpz_class>& v) {
    for (const auto& x : v) if (x != 0) return false;
    return true;
}

}  // namespace

BasisRefreshResult lll_refresh(
    const Basis& current,
    const std::vector<std::int64_t>& candidate_coefficients,
    double lll_delta,
    double potential_rel_tolerance) {
    BasisRefreshResult result;
    result.basis = current;
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
        if (is_zero_vector(candidate)) {
            result.error = "basis refresh candidate is the zero vector";
            return result;
        }

        GsoData before_gso;
        before_gso.recompute(current);
        result.potential_before = before_gso.log_potential();
        const mpz_class determinant_before = current.exact_abs_determinant();

        Matrix expanded(n + 1, cols);
        for (int j = 0; j < cols; ++j) {
            mpz_set(expanded[0][j].get_data(), candidate[static_cast<std::size_t>(j)].get_mpz_t());
        }
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < cols; ++j) expanded[i + 1][j] = current.matrix()[i][j];
        }

        fplll::lll_reduction(expanded, lll_delta);

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
        Basis candidate_basis(std::move(reduced));
        if (candidate_basis.exact_abs_determinant() != determinant_before) {
            result.error = "refresh determinant mismatch; transaction rolled back";
            return result;
        }

        GsoData after_gso;
        after_gso.recompute(candidate_basis);
        result.potential_after = after_gso.log_potential();
        const double scale = std::max(1.0, std::fabs(result.potential_before));
        const double tolerance = std::max(0.0, potential_rel_tolerance) * scale;
        result.accepted = result.potential_after <= result.potential_before + tolerance;
        result.completed = true;
        if (!result.accepted) {
            result.error = "refresh rejected by non-worsening GSO-potential transaction gate";
            return result;
        }

        result.changed = !equal_basis(current.matrix(), candidate_basis.matrix());
        result.basis = std::move(candidate_basis);
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
