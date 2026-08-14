#include "mcts/basis.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace mcts_enum {
namespace {

void append_u32_be(std::string& out, std::uint32_t value) {
    out.push_back(static_cast<char>((value >> 24) & 0xff));
    out.push_back(static_cast<char>((value >> 16) & 0xff));
    out.push_back(static_cast<char>((value >> 8) & 0xff));
    out.push_back(static_cast<char>(value & 0xff));
}

std::uint32_t read_u32_be(const std::string& data, std::size_t& offset) {
    if (offset + 4 > data.size()) throw std::runtime_error("truncated MCTSBAS1 packet");
    const auto* p = reinterpret_cast<const unsigned char*>(data.data() + offset);
    offset += 4;
    return (static_cast<std::uint32_t>(p[0]) << 24) |
           (static_cast<std::uint32_t>(p[1]) << 16) |
           (static_cast<std::uint32_t>(p[2]) << 8) |
           static_cast<std::uint32_t>(p[3]);
}

}  // namespace

Basis::Basis(Matrix matrix) : matrix_(std::move(matrix)) {
    if (matrix_.get_rows() <= 0 || matrix_.get_cols() <= 0) {
        throw std::runtime_error("basis must be non-empty");
    }
    if (matrix_.get_rows() != matrix_.get_cols()) {
        throw std::runtime_error("MCTS v0.1 requires a square full-rank basis");
    }
}

Basis Basis::from_text(const std::string& text) {
    Matrix matrix;
    std::istringstream input(text);
    input >> matrix;
    if (!input && matrix.get_rows() == 0) {
        throw std::runtime_error("failed to parse lattice basis");
    }
    return Basis(std::move(matrix));
}

Basis Basis::from_packet(const std::string& packet) {
    if (packet.size() < 16 || std::memcmp(packet.data(), "MCTSBAS1", 8) != 0) {
        throw std::runtime_error("invalid in-memory basis packet magic");
    }
    std::size_t offset = 8;
    const std::uint32_t rows = read_u32_be(packet, offset);
    const std::uint32_t cols = read_u32_be(packet, offset);
    if (rows == 0 || cols == 0 || rows != cols) {
        throw std::runtime_error("invalid MCTSBAS1 dimensions");
    }

    Matrix matrix(static_cast<int>(rows), static_cast<int>(cols));
    for (std::uint32_t i = 0; i < rows; ++i) {
        for (std::uint32_t j = 0; j < cols; ++j) {
            if (offset >= packet.size()) throw std::runtime_error("truncated MCTSBAS1 sign");
            const std::int8_t sign = static_cast<std::int8_t>(packet[offset++]);
            const std::uint32_t size = read_u32_be(packet, offset);
            if (offset + size > packet.size()) throw std::runtime_error("truncated MCTSBAS1 integer");
            if (sign == 0) {
                if (size != 0) throw std::runtime_error("zero integer with non-zero magnitude");
                matrix[i][j] = 0;
            } else {
                mpz_import(matrix[i][j].get_data(), size, 1, 1, 1, 0, packet.data() + offset);
                if (sign < 0) mpz_neg(matrix[i][j].get_data(), matrix[i][j].get_data());
            }
            offset += size;
        }
    }
    if (offset != packet.size()) throw std::runtime_error("trailing bytes in MCTSBAS1 packet");
    return Basis(std::move(matrix));
}

std::string Basis::to_text() const {
    std::ostringstream output;
    output << matrix_;
    return output.str();
}

std::string Basis::to_packet() const {
    std::string out;
    out.reserve(static_cast<std::size_t>(dimension()) * columns() * 16 + 16);
    out.append("MCTSBAS1", 8);
    append_u32_be(out, static_cast<std::uint32_t>(dimension()));
    append_u32_be(out, static_cast<std::uint32_t>(columns()));

    for (int i = 0; i < dimension(); ++i) {
        for (int j = 0; j < columns(); ++j) {
            mpz_srcptr value = matrix_[i][j].get_data();
            const int sign = mpz_sgn(value);
            out.push_back(static_cast<char>(sign < 0 ? 0xff : sign > 0 ? 0x01 : 0x00));
            if (sign == 0) {
                append_u32_be(out, 0);
                continue;
            }
            const std::size_t capacity = (mpz_sizeinbase(value, 2) + 7) / 8;
            std::string magnitude(capacity, '\0');
            std::size_t written = 0;
            mpz_export(magnitude.data(), &written, 1, 1, 1, 0, value);
            magnitude.resize(written);
            append_u32_be(out, static_cast<std::uint32_t>(written));
            out.append(magnitude);
        }
    }
    return out;
}

std::vector<mpz_class> Basis::exact_vector(const std::vector<std::int64_t>& z) const {
    if (static_cast<int>(z.size()) != dimension()) {
        throw std::runtime_error("coefficient vector dimension mismatch");
    }
    static_assert(sizeof(long) >= sizeof(std::int64_t), "64-bit long is required");
    std::vector<mpz_class> vector(columns(), 0);
    for (int i = 0; i < dimension(); ++i) {
        if (z[i] == 0) continue;
        mpz_class coefficient(static_cast<long>(z[i]));
        for (int j = 0; j < columns(); ++j) {
            mpz_class entry;
            mpz_set(entry.get_mpz_t(), matrix_[i][j].get_data());
            vector[j] += coefficient * entry;
        }
    }
    return vector;
}

mpz_class Basis::exact_squared_norm(const std::vector<std::int64_t>& z) const {
    const auto vector = exact_vector(z);
    mpz_class norm = 0;
    for (const auto& x : vector) norm += x * x;
    return norm;
}

mpz_class Basis::exact_abs_determinant() const {
    const int n = dimension();
    if (n != columns()) {
        throw std::runtime_error("exact determinant requires a square basis");
    }
    if (n == 0) return 1;

    std::vector<std::vector<mpz_class>> a(
        static_cast<std::size_t>(n),
        std::vector<mpz_class>(static_cast<std::size_t>(n)));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            mpz_set(a[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)].get_mpz_t(),
                    matrix_[i][j].get_data());
        }
    }

    mpz_class previous = 1;
    int sign = 1;
    for (int k = 0; k < n - 1; ++k) {
        int pivot = k;
        while (pivot < n && a[static_cast<std::size_t>(pivot)][static_cast<std::size_t>(k)] == 0) {
            ++pivot;
        }
        if (pivot == n) return 0;
        if (pivot != k) {
            std::swap(a[static_cast<std::size_t>(pivot)], a[static_cast<std::size_t>(k)]);
            sign = -sign;
        }

        const mpz_class pivot_value = a[static_cast<std::size_t>(k)][static_cast<std::size_t>(k)];
        for (int i = k + 1; i < n; ++i) {
            for (int j = k + 1; j < n; ++j) {
                mpz_class numerator =
                    a[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] * pivot_value -
                    a[static_cast<std::size_t>(i)][static_cast<std::size_t>(k)] *
                        a[static_cast<std::size_t>(k)][static_cast<std::size_t>(j)];
                if (k > 0) {
                    mpz_divexact(numerator.get_mpz_t(), numerator.get_mpz_t(), previous.get_mpz_t());
                }
                a[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = std::move(numerator);
            }
        }
        previous = pivot_value;
    }

    mpz_class det = a[static_cast<std::size_t>(n - 1)][static_cast<std::size_t>(n - 1)];
    if (sign < 0) det = -det;
    if (det < 0) det = -det;
    return det;
}

mpz_class Basis::first_vector_squared_norm() const {
    mpz_class norm = 0;
    for (int j = 0; j < columns(); ++j) {
        mpz_class x;
        mpz_set(x.get_mpz_t(), matrix_[0][j].get_data());
        norm += x * x;
    }
    return norm;
}

std::vector<std::int64_t> Basis::first_vector_coefficients() const {
    std::vector<std::int64_t> z(dimension(), 0);
    if (!z.empty()) z[0] = 1;
    return z;
}

long double Basis::log_positive_mpz(const mpz_class& value) {
    if (value <= 0) throw std::runtime_error("log_positive_mpz requires value > 0");
    long exponent = 0;
    const double mantissa = mpz_get_d_2exp(&exponent, value.get_mpz_t());
    return std::log(std::fabs(static_cast<long double>(mantissa))) +
           static_cast<long double>(exponent) * std::log(2.0L);
}

long double Basis::scaled_positive_mpz(const mpz_class& value, long scale_exp2) {
    if (value <= 0) return 0.0L;
    long exponent = 0;
    const double mantissa = mpz_get_d_2exp(&exponent, value.get_mpz_t());
    return std::ldexp(static_cast<long double>(mantissa), exponent - scale_exp2);
}

}  // namespace mcts_enum
