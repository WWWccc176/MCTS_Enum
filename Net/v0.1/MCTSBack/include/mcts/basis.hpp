#pragma once

#include <fplll.h>
#include <gmpxx.h>

#include <cstdint>
#include <string>
#include <vector>

namespace mcts_enum {

using Matrix = fplll::ZZ_mat<mpz_t>;

class Basis {
public:
    Basis() = default;
    explicit Basis(Matrix matrix);

    static Basis from_text(const std::string& text);
    static Basis from_packet(const std::string& packet);

    std::string to_text() const;
    std::string to_packet() const;

    int dimension() const noexcept { return matrix_.get_rows(); }
    int columns() const noexcept { return matrix_.get_cols(); }
    const Matrix& matrix() const noexcept { return matrix_; }
    Matrix& mutable_matrix() noexcept { return matrix_; }

    std::vector<mpz_class> exact_vector(const std::vector<std::int64_t>& z) const;
    mpz_class exact_squared_norm(const std::vector<std::int64_t>& z) const;
    mpz_class first_vector_squared_norm() const;
    mpz_class exact_abs_determinant() const;
    std::vector<std::int64_t> first_vector_coefficients() const;

    static long double log_positive_mpz(const mpz_class& value);
    static long double scaled_positive_mpz(const mpz_class& value, long scale_exp2);

private:
    Matrix matrix_;
};

}  // namespace mcts_enum
