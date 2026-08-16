#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <fplll.h>

#include <gmpxx.h>
#include <cstring>
#include <sstream>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "types.hpp"
#include "matrix_utils.hpp"
#include "bkz2_engine.hpp"
#include "extreme_reducer.hpp"
#include "sieve_bridge.hpp"
#include "../cuda/lattice_cuda.h"

namespace py = pybind11;
using lattice_backend::Matrix;

namespace {

std::unordered_map<int64_t, Matrix> g_matrix_pool;
int64_t g_next_matrix_id = 1;
std::atomic<int64_t> g_next_serial_task_id{-1};
std::mutex g_matrix_pool_mutex;

constexpr int kAdaptiveSieveThreshold = lattice_backend::kSieveThreshold;
constexpr int kCudaFeatureMinDimension = 256;

double norm_from_log(double log_norm) {
    if (!std::isfinite(log_norm)) {
        return log_norm < 0.0 ? 0.0 : std::numeric_limits<double>::infinity();
    }
    if (log_norm >= std::log(std::numeric_limits<double>::max())) {
        return std::numeric_limits<double>::infinity();
    }
    if (log_norm <= std::log(std::numeric_limits<double>::min())) {
        return 0.0;
    }
    return std::exp(log_norm);
}

py::dict reduce_result_to_dict(const lattice_backend::ReduceResult& result) {
    py::dict out;
    out["backend"] = result.backend;
    out["completed"] = result.completed;
    out["changed"] = result.changed;
    out["exact"] = result.exact;
    out["non_worsening"] = result.non_worsening;
    out["accepted"] = result.accepted;
    out["early_stopped"] = result.early_stopped;
    out["quality_target_reached"] = result.quality_target_reached;
    out["stop_reason"] = lattice_backend::to_string(result.stop_reason);
    out["error"] = result.error;
    out["actual_beta"] = result.actual_beta;
    out["enumeration_rounds"] = result.enumeration_rounds;
    out["potential_drop_per_dimension"] = result.potential_drop_per_dimension;
    out["first_gso_log_drop"] = result.first_gso_log_drop;
    out["sieve_dimension"] = result.sieve_dimension;
    out["dimension_for_free"] = result.dimension_for_free;
    out["bgj_calls"] = result.bgj_calls;
    out["database_vectors"] = result.database_vectors;
    out["sieve_batches"] = result.sieve_batches;
    out["sieve_solution_vectors"] = result.sieve_solution_vectors;
    out["sieve_best_score"] = result.sieve_best_score;
    out["sieve_elapsed_seconds"] = result.sieve_elapsed_seconds;
    out["sieve_idle_seconds"] = result.sieve_idle_seconds;
    out["time_ms"] = result.time_ms;
    return out;
}

int64_t create_matrix(const std::string& text) {
    Matrix matrix = lattice_backend::parse_matrix(text);
    std::lock_guard<std::mutex> lock(g_matrix_pool_mutex);
    const int64_t id = g_next_matrix_id++;
    g_matrix_pool.emplace(id, std::move(matrix));
    return id;
}

int64_t create_matrix_lll(const std::string& text) {
    Matrix matrix = lattice_backend::parse_matrix(text);
    {
        py::gil_scoped_release release;
        fplll::lll_reduction(matrix, 0.999);
    }
    std::lock_guard<std::mutex> lock(g_matrix_pool_mutex);
    const int64_t id = g_next_matrix_id++;
    g_matrix_pool.emplace(id, std::move(matrix));
    return id;
}

std::string dump_matrix_api(int64_t id) {
    std::lock_guard<std::mutex> lock(g_matrix_pool_mutex);
    const auto iterator = g_matrix_pool.find(id);
    if (iterator == g_matrix_pool.end()) throw std::runtime_error("invalid matrix id");
    return lattice_backend::dump_matrix(iterator->second);
}

void free_matrix(int64_t id) {
    std::lock_guard<std::mutex> lock(g_matrix_pool_mutex);
    g_matrix_pool.erase(id);
}

int64_t clone_matrix(int64_t id) {
    std::lock_guard<std::mutex> lock(g_matrix_pool_mutex);
    const auto iterator = g_matrix_pool.find(id);
    if (iterator == g_matrix_pool.end()) return -1;
    const int64_t new_id = g_next_matrix_id++;
    g_matrix_pool.emplace(new_id, iterator->second);
    return new_id;
}

py::dict evaluate_matrix(int64_t id) {
    std::vector<double> matrix;
    std::vector<double> scales;
    std::vector<double> gso;
    std::vector<double> gram;
    std::vector<float> cosine;
    double statistics[2] = {0.0, 0.0};
    int dimension = 0;
    int cols = 0;
    bool gpu = false;

    {
        py::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(g_matrix_pool_mutex);
        const auto iterator = g_matrix_pool.find(id);
        if (iterator == g_matrix_pool.end()) throw std::runtime_error("invalid matrix id");

        lattice_backend::extract_scaled_matrix(
            iterator->second, matrix, scales, dimension, cols);
        lattice_backend::gso_log_norms(iterator->second, gso);

        gram.assign(static_cast<std::size_t>(dimension) * dimension, 0.0);
        cosine.assign(static_cast<std::size_t>(dimension) * dimension, 0.0f);
#ifdef USE_CUDA
        // Small A11 states are faster on CPU and should not instantiate a CUDA
        // context in every environment process.
        if (dimension >= kCudaFeatureMinDimension) {
            gpu = cuda_gram_cosine(matrix.data(), dimension, cols,
                                   gram.data(), cosine.data(), statistics);
        }
#endif
        if (!gpu) {
            for (int i = 0; i < dimension; ++i) {
                for (int j = 0; j <= i; ++j) {
                    double product = 0.0;
                    for (int k = 0; k < cols; ++k) {
                        product += matrix[static_cast<std::size_t>(i) * cols + k] *
                                   matrix[static_cast<std::size_t>(j) * cols + k];
                    }
                    gram[static_cast<std::size_t>(i) * dimension + j] = product;
                    gram[static_cast<std::size_t>(j) * dimension + i] = product;
                }
            }
            for (int i = 0; i < dimension; ++i) {
                for (int j = 0; j < i; ++j) {
                    const double denominator = std::sqrt(
                        gram[static_cast<std::size_t>(i) * dimension + i] *
                        gram[static_cast<std::size_t>(j) * dimension + j]) + 1e-20;
                    const double value = std::fabs(
                        gram[static_cast<std::size_t>(i) * dimension + j] /
                        denominator);
                    cosine[static_cast<std::size_t>(i) * dimension + j] =
                        static_cast<float>(value);
                    statistics[0] = std::max(statistics[0], value);
                    statistics[1] += value;
                }
            }
        }
    }

    py::array_t<double> gso_array({dimension});
    std::copy(gso.begin(), gso.end(), gso_array.mutable_data());
    py::array_t<float> cosine_array({dimension, dimension});
    std::copy(cosine.begin(), cosine.end(), cosine_array.mutable_data());

    py::dict out;
    out["gs_log_norms"] = gso_array;
    out["cos_matrix"] = cosine_array;
    out["max_cos"] = statistics[0];
    out["mean_cos"] = dimension > 1
                           ? statistics[1] /
                                 (static_cast<double>(dimension) * (dimension - 1) / 2.0)
                           : 0.0;
    out["feature_device"] = gpu ? "cuda" : "cpu";
    return out;
}

int minimum_action_beta_api() {
    return lattice_backend::kMinimumActionBeta;
}

int adaptive_sieve_threshold_api() {
    return kAdaptiveSieveThreshold;
}

bool action_uses_gpu_api(int64_t matrix_id, int pos, int beta) {
    std::lock_guard<std::mutex> lock(g_matrix_pool_mutex);
    const auto iterator = g_matrix_pool.find(matrix_id);
    if (iterator == g_matrix_pool.end()) throw std::runtime_error("invalid matrix id");

    const int rows = iterator->second.get_rows();
    if (pos < 0 || pos >= rows ||
        beta < lattice_backend::kMinimumActionBeta ||
        beta > lattice_backend::kMaximumActionBeta) {
        throw std::runtime_error("invalid adaptive block");
    }
    const int actual_beta = std::min(beta, rows - pos);
    if (actual_beta < lattice_backend::kMinimumActionBeta) {
        throw std::runtime_error("clipped adaptive block is smaller than beta=21");
    }
    return actual_beta >= kAdaptiveSieveThreshold;
}

py::dict reduce_extreme_api(int64_t matrix_id, int pos, int beta,
                            bool use_sieve) {
    lattice_backend::ReduceResult result;
    {
        py::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(g_matrix_pool_mutex);
        const auto iterator = g_matrix_pool.find(matrix_id);
        if (iterator == g_matrix_pool.end()) throw std::runtime_error("invalid matrix id");
        result = lattice_backend::reduce_extreme(
            matrix_id, iterator->second, pos, beta, use_sieve);
    }
    return reduce_result_to_dict(result);
}

py::dict reduce_bkz2_global_api(int64_t matrix_id, int beta, int loops) {
    lattice_backend::ReduceResult result;
    result.backend = "bkz2_global";
    const auto started = std::chrono::steady_clock::now();

    {
        py::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(g_matrix_pool_mutex);
        const auto iterator = g_matrix_pool.find(matrix_id);
        if (iterator == g_matrix_pool.end()) throw std::runtime_error("invalid matrix id");

        Matrix before = iterator->second;
        result.actual_beta = std::max(2, std::min(beta, iterator->second.get_rows()));
        result.completed = lattice_backend::run_bkz2(
            iterator->second, result.actual_beta, std::max(1, loops), 1.0,
            &result.error);
        result.exact = true;
        result.changed = result.completed &&
                         !lattice_backend::matrices_equal(before, iterator->second);
        result.non_worsening = result.completed &&
            lattice_backend::log_potential(iterator->second) <=
                lattice_backend::log_potential(before) + 1e-10;
        result.accepted = result.changed && result.non_worsening;
        if (!result.non_worsening) iterator->second = std::move(before);
        result.stop_reason = !result.completed
                                 ? lattice_backend::StopReason::precondition_failed
                                 : result.accepted
                                       ? lattice_backend::StopReason::completed
                                       : result.changed && !result.non_worsening
                                             ? lattice_backend::StopReason::non_worsening_rejected
                                             : lattice_backend::StopReason::no_change;
    }

    result.time_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    return reduce_result_to_dict(result);
}

std::string extract_block_api(int64_t matrix_id, int pos, int beta) {
    std::lock_guard<std::mutex> lock(g_matrix_pool_mutex);
    const auto iterator = g_matrix_pool.find(matrix_id);
    if (iterator == g_matrix_pool.end()) {
        throw std::runtime_error("invalid matrix id");
    }

    const Matrix& basis = iterator->second;
    if (pos < 0 || beta <= 0 || pos + beta > basis.get_rows()) {
        throw std::runtime_error("invalid exact block extraction range");
    }

    return lattice_backend::dump_matrix(
        lattice_backend::copy_block(basis, pos, beta));
}

py::dict sieve_reduce_serialized_api(
    const std::string& matrix_text,
    int beta,
    int max_candidates,
    int max_rounds,
    int64_t max_pairs,
    double time_budget_s,
    int memory_budget_mb,
    double min_b1_rel_improvement,
    double min_logpot_improvement,
    int free_dim,
    int free_dim_cap) {
    Matrix block = lattice_backend::parse_matrix(matrix_text);
    if (beta != block.get_rows() || beta < lattice_backend::kMinimumActionBeta ||
        beta > lattice_backend::kMaximumActionBeta ||
        block.get_cols() < block.get_rows()) {
        throw std::runtime_error(
            "serialized BGJ block must be full-row-rank shaped within the project action range");
    }

    const Matrix original = block;
    const int64_t task_id =
        g_next_serial_task_id.fetch_sub(1, std::memory_order_relaxed);
    lattice_backend::ReduceResult result;
    result.backend = "persistent_bgj_sieve_worker";
    result.actual_beta = beta;

    double potential_before = std::numeric_limits<double>::quiet_NaN();
    double potential_after = std::numeric_limits<double>::quiet_NaN();
    double log_b1_before = std::numeric_limits<double>::quiet_NaN();
    double log_b1_after = std::numeric_limits<double>::quiet_NaN();
    double b1_relative_improvement = 0.0;
    double logpot_improvement = 0.0;
    const auto started = std::chrono::steady_clock::now();

    {
        py::gil_scoped_release release;

        potential_before = lattice_backend::log_potential(original);
        log_b1_before = lattice_backend::first_gso_log_norm(original);

        if (!lattice_backend::run_bkz2_preconditioner(
                block, lattice_backend::PreconditionerProfile::normal,
                &result.error)) {
            result.stop_reason = lattice_backend::StopReason::precondition_failed;
        } else {
            lattice_backend::SieveBudget budget =
                lattice_backend::sieve_budget_from_environment();

            // One complete BGJ invocation is the action contract. The Python
            // argument is retained for API compatibility but cannot increase
            // this limit.
            budget.max_bgj_calls = 1;
            budget.progressive = false;
            double remaining_budget_seconds = time_budget_s;
            if (std::isfinite(time_budget_s) && time_budget_s > 0.0) {
                const double elapsed_before_sieve =
                    std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - started
                    ).count();
                remaining_budget_seconds = time_budget_s - elapsed_before_sieve;
            }

            if (std::isfinite(time_budget_s) &&
                time_budget_s > 0.0 &&
                remaining_budget_seconds < 1.0) {
                result.stop_reason = lattice_backend::StopReason::budget_exhausted;
                result.early_stopped = true;
                result.error =
                    "BGJ action budget was exhausted during BKZ2 preconditioning";
            } else {
                if (std::isfinite(remaining_budget_seconds) &&
                    remaining_budget_seconds > 0.0) {
                    budget.max_wall_seconds = remaining_budget_seconds;
                    budget.bgj_max_seconds = remaining_budget_seconds;
                }

                const lattice_backend::SieveRunInfo sieve =
                    lattice_backend::run_local_extreme_sieve(
                        block, task_id, 0, budget);
                result.completed = sieve.completed;
                result.exact = sieve.exact;
                result.stop_reason = sieve.stop_reason;
                result.early_stopped =
                    sieve.stop_reason == lattice_backend::StopReason::budget_exhausted ||
                    sieve.stop_reason == lattice_backend::StopReason::stagnation ||
                    sieve.stop_reason == lattice_backend::StopReason::quality_target_reached;
                result.error = sieve.error;
                result.sieve_dimension = sieve.final_csd;
                result.dimension_for_free = sieve.dimension_for_free;
                result.bgj_calls = sieve.bgj_calls;
                result.database_vectors = sieve.vectors;
                result.sieve_batches = sieve.batches;
                result.sieve_solution_vectors = sieve.solution_vectors;
                result.sieve_best_score = sieve.best_score;
                result.sieve_elapsed_seconds = sieve.elapsed_seconds;
                result.sieve_idle_seconds = sieve.idle_seconds;
            }

            // Exact post-BKZ/LLL is deliberately performed only by
            // apply_external_block() in the process that owns the full MPZ
            // basis. The GPU worker returns the exactly recovered sieve block.
        }

        potential_after = lattice_backend::log_potential(block);
        log_b1_after = lattice_backend::first_gso_log_norm(block);
        logpot_improvement = potential_before - potential_after;
        const double exponent = std::max(
            -700.0, std::min(700.0, log_b1_after - log_b1_before));
        b1_relative_improvement = 1.0 - std::exp(exponent);

        result.changed = result.completed && result.exact &&
                         !lattice_backend::matrices_equal(original, block);
        result.non_worsening = result.completed && result.exact &&
                               std::isfinite(potential_after) &&
                               potential_after <= potential_before + 1e-10;
        result.accepted = result.changed && result.non_worsening;
        result.quality_target_reached = result.completed && result.exact &&
            (b1_relative_improvement >= min_b1_rel_improvement ||
             logpot_improvement >= min_logpot_improvement);

        if (result.completed && !result.changed &&
            (result.stop_reason == lattice_backend::StopReason::none ||
             result.stop_reason == lattice_backend::StopReason::completed)) {
            result.stop_reason = lattice_backend::StopReason::no_change;
        } else if (result.completed && result.changed &&
                   !result.non_worsening) {
            result.stop_reason =
                lattice_backend::StopReason::non_worsening_rejected;
        } else if (result.accepted &&
                   result.stop_reason == lattice_backend::StopReason::none) {
            result.stop_reason = lattice_backend::StopReason::completed;
        }
    }

    result.time_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();

    py::dict out = reduce_result_to_dict(result);
    const std::string serialized_block = lattice_backend::dump_matrix(block);
    out["block_matrix"] = serialized_block;
    out["matrix"] = serialized_block;
    out["exact_recovery"] = result.exact;
    out["potential_before"] = potential_before;
    out["potential_after"] = potential_after;
    out["log_b1_before"] = log_b1_before;
    out["log_b1_after"] = log_b1_after;
    out["b1_relative_improvement"] = b1_relative_improvement;
    out["logpot_improvement"] = logpot_improvement;
    out["requested_max_candidates"] = max_candidates;
    out["requested_max_rounds"] = max_rounds;
    out["requested_max_pairs"] = max_pairs;
    out["requested_memory_budget_mb"] = memory_budget_mb;
    out["requested_free_dim"] = free_dim;
    out["requested_free_dim_cap"] = free_dim_cap;
    out["effective_max_bgj_calls"] = 1;
    return out;
}

py::dict apply_external_block_api(
    int64_t matrix_id,
    int pos,
    const std::string& block_text,
    int post_bkz_loops,
    bool run_full_lll_after) {
    Matrix candidate = lattice_backend::parse_matrix(block_text);
    lattice_backend::ReduceResult result;
    result.backend = "apply_external_exact_block";
    result.actual_beta = candidate.get_rows();
    const auto started = std::chrono::steady_clock::now();

    double potential_before = std::numeric_limits<double>::quiet_NaN();
    double potential_after = std::numeric_limits<double>::quiet_NaN();
    double log_b1_before = std::numeric_limits<double>::quiet_NaN();
    double log_b1_after = std::numeric_limits<double>::quiet_NaN();

    {
        py::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(g_matrix_pool_mutex);
        const auto iterator = g_matrix_pool.find(matrix_id);
        if (iterator == g_matrix_pool.end()) {
            throw std::runtime_error("invalid matrix id");
        }

        Matrix& basis = iterator->second;
        if (candidate.get_rows() <= 0 || pos < 0 ||
            pos + candidate.get_rows() > basis.get_rows() ||
            candidate.get_cols() != basis.get_cols()) {
            result.stop_reason = lattice_backend::StopReason::invalid_input;
            result.error = "external exact block shape or position mismatch";
        } else {
            Matrix full_before = basis;
            const Matrix local_before = lattice_backend::copy_block(
                basis, pos, candidate.get_rows());
            potential_before = lattice_backend::log_potential(full_before);
            log_b1_before = lattice_backend::first_gso_log_norm(full_before);

            result.exact = lattice_backend::validate_exact_unimodular_basis(
                local_before, candidate, &result.error);
            if (!result.exact) {
                result.stop_reason =
                    lattice_backend::StopReason::exact_recovery_failed;
            } else {
                if (post_bkz_loops > 0) {
                    if (!lattice_backend::run_bkz2(
                            candidate,
                            std::min(30, candidate.get_rows()),
                            post_bkz_loops,
                            1.0,
                            &result.error)) {
                        result.stop_reason =
                            lattice_backend::StopReason::precondition_failed;
                    }
                }

                if (result.stop_reason == lattice_backend::StopReason::none) {
                    fplll::lll_reduction(candidate, 0.999);

                    if (!lattice_backend::validate_exact_unimodular_basis(
                            local_before, candidate, &result.error)) {
                        result.exact = false;
                        result.stop_reason =
                            lattice_backend::StopReason::exact_recovery_failed;
                    } else if (!lattice_backend::write_block(
                                   basis, pos, candidate, &result.error)) {
                        result.stop_reason =
                            lattice_backend::StopReason::invalid_input;
                    } else {
                        if (run_full_lll_after) {
                            fplll::lll_reduction(basis, 0.999);
                        }

                        result.completed = true;
                        potential_after = lattice_backend::log_potential(basis);
                        log_b1_after =
                            lattice_backend::first_gso_log_norm(basis);
                        result.changed = !lattice_backend::matrices_equal(
                            full_before, basis);
                        result.non_worsening =
                            std::isfinite(potential_after) &&
                            potential_after <= potential_before + 1e-10;
                        result.accepted =
                            result.changed && result.non_worsening;

                        if (!result.non_worsening) {
                            basis = std::move(full_before);
                            result.accepted = false;
                            result.stop_reason = lattice_backend::StopReason::
                                non_worsening_rejected;
                            if (result.error.empty()) {
                                result.error =
                                    "external block failed the full-basis transaction gate";
                            }
                        } else {
                            result.stop_reason = result.changed
                                ? lattice_backend::StopReason::completed
                                : lattice_backend::StopReason::no_change;
                        }
                    }
                }
            }

            if (!result.completed) {
                potential_after = potential_before;
                log_b1_after = log_b1_before;
            }
        }
    }

    result.time_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    py::dict out = reduce_result_to_dict(result);
    out["pot_before"] = potential_before;
    out["pot_after"] = potential_after;
    out["b1_before"] = norm_from_log(log_b1_before);
    out["b1_after"] = norm_from_log(log_b1_after);
    out["log_b1_before"] = log_b1_before;
    out["log_b1_after"] = log_b1_after;
    return out;
}

py::dict full_lll_api(int64_t matrix_id) {
    lattice_backend::ReduceResult result;
    result.backend = "full_lll";
    const auto started = std::chrono::steady_clock::now();
    double potential_before = std::numeric_limits<double>::quiet_NaN();
    double potential_after = std::numeric_limits<double>::quiet_NaN();
    double log_b1_before = std::numeric_limits<double>::quiet_NaN();
    double log_b1_after = std::numeric_limits<double>::quiet_NaN();

    {
        py::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(g_matrix_pool_mutex);
        const auto iterator = g_matrix_pool.find(matrix_id);
        if (iterator == g_matrix_pool.end()) {
            throw std::runtime_error("invalid matrix id");
        }

        Matrix& basis = iterator->second;
        Matrix before = basis;
        result.actual_beta = basis.get_rows();
        potential_before = lattice_backend::log_potential(before);
        log_b1_before = lattice_backend::first_gso_log_norm(before);

        try {
            fplll::lll_reduction(basis, 0.999);
            result.completed = true;
            result.exact = true;
            potential_after = lattice_backend::log_potential(basis);
            log_b1_after = lattice_backend::first_gso_log_norm(basis);
            result.changed = !lattice_backend::matrices_equal(before, basis);
            result.non_worsening = std::isfinite(potential_after) &&
                                   potential_after <= potential_before + 1e-10;
            result.accepted = result.changed && result.non_worsening;

            if (!result.non_worsening) {
                basis = std::move(before);
                potential_after = potential_before;
                log_b1_after = log_b1_before;
                result.accepted = false;
                result.stop_reason =
                    lattice_backend::StopReason::non_worsening_rejected;
                result.error = "full LLL failed the full-basis transaction gate";
            } else {
                result.stop_reason = result.changed
                    ? lattice_backend::StopReason::completed
                    : lattice_backend::StopReason::no_change;
            }
        } catch (const std::exception& ex) {
            basis = std::move(before);
            result.stop_reason = lattice_backend::StopReason::exception;
            result.error = ex.what();
            potential_after = potential_before;
            log_b1_after = log_b1_before;
        } catch (...) {
            basis = std::move(before);
            result.stop_reason = lattice_backend::StopReason::exception;
            result.error = "full LLL failed with an unknown exception";
            potential_after = potential_before;
            log_b1_after = log_b1_before;
        }
    }

    result.time_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    py::dict out = reduce_result_to_dict(result);
    out["pot_before"] = potential_before;
    out["pot_after"] = potential_after;
    out["b1_before"] = norm_from_log(log_b1_before);
    out["b1_after"] = norm_from_log(log_b1_after);
    out["log_b1_before"] = log_b1_before;
    out["log_b1_after"] = log_b1_after;
    return out;
}

py::dict reduce_sieve_block_api(int64_t matrix_id, int pos, int beta) {
    lattice_backend::ReduceResult result;
    result.backend = "local_bgj_sieve_final";
    const auto started = std::chrono::steady_clock::now();

    {
        py::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(g_matrix_pool_mutex);
        const auto iterator = g_matrix_pool.find(matrix_id);
        if (iterator == g_matrix_pool.end()) throw std::runtime_error("invalid matrix id");

        Matrix& basis = iterator->second;
        if (pos < 0 || pos >= basis.get_rows() || beta < 2) {
            throw std::runtime_error("invalid sieve block");
        }

        Matrix full_before = basis;
        const double full_potential_before = lattice_backend::log_potential(full_before);
        result.actual_beta = std::min(beta, basis.get_rows() - pos);
        Matrix local_before = lattice_backend::copy_block(basis, pos, result.actual_beta);
        Matrix candidate = local_before;
        const double local_potential_before = lattice_backend::log_potential(local_before);

        const lattice_backend::SieveRunInfo sieve =
            lattice_backend::run_local_extreme_sieve(candidate, matrix_id, pos);
        result.completed = sieve.completed;
        result.exact = sieve.exact;
        result.stop_reason = sieve.stop_reason;
        result.error = sieve.error;
        result.sieve_dimension = sieve.final_csd;
        result.dimension_for_free = sieve.dimension_for_free;
        result.bgj_calls = sieve.bgj_calls;
        result.database_vectors = sieve.vectors;

        if (result.completed && result.exact) {
            fplll::lll_reduction(candidate, 0.999);
            const double local_potential_after = lattice_backend::log_potential(candidate);
            const bool local_non_worsening =
                std::isfinite(local_potential_after) &&
                local_potential_after <= local_potential_before + 1e-10;

            if (local_non_worsening &&
                !lattice_backend::matrices_equal(local_before, candidate)) {
                if (!lattice_backend::write_block(basis, pos, candidate, &result.error)) {
                    basis = std::move(full_before);
                    result.completed = false;
                    result.stop_reason = lattice_backend::StopReason::invalid_input;
                }
            }

            if (result.completed) {
                fplll::lll_reduction(basis, 0.999);
                const double full_potential_after = lattice_backend::log_potential(basis);
                result.non_worsening =
                    std::isfinite(full_potential_after) &&
                    full_potential_after <= full_potential_before + 1e-10;
                result.changed =
                    !lattice_backend::matrices_equal(full_before, basis);
                result.accepted = result.changed && result.non_worsening;

                if (!result.non_worsening) {
                    basis = std::move(full_before);
                    result.accepted = false;
                    result.stop_reason =
                        lattice_backend::StopReason::non_worsening_rejected;
                    if (result.error.empty()) {
                        result.error =
                            "cycle-tail sieve/LLL result failed the full-basis transaction gate";
                    }
                } else {
                    result.stop_reason = result.changed
                        ? lattice_backend::StopReason::completed
                        : lattice_backend::StopReason::no_change;
                }
            }
        }
    }

    result.time_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    return reduce_result_to_dict(result);
}

py::dict reduce_serialized_block_api(const std::string& matrix_text,
                                     bool use_sieve) {
    Matrix block = lattice_backend::parse_matrix(matrix_text);
    const int64_t task_id =
        g_next_serial_task_id.fetch_sub(1, std::memory_order_relaxed);
    lattice_backend::ReduceResult result;
    {
        py::gil_scoped_release release;
        result = lattice_backend::reduce_extreme(
            task_id, block, 0, block.get_rows(), use_sieve);
    }
    py::dict out = reduce_result_to_dict(result);
    out["matrix"] = lattice_backend::dump_matrix(block);
    return out;
}

py::dict reduce_compat(int64_t matrix_id, const std::string& method,
                       int parameter, int position) {
    const bool use_sieve = method == "ORACLE" || method == "SIEVE" ||
                           method == "EXTREME";
    return reduce_extreme_api(matrix_id, position, parameter, use_sieve);
}



std::uint64_t read_u64_be(const unsigned char* p) {
    std::uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v = (v << 8) | p[i];
    return v;
}

void append_u32_be(std::string& out, std::uint32_t value) {
    out.push_back(static_cast<char>((value >> 24) & 0xff));
    out.push_back(static_cast<char>((value >> 16) & 0xff));
    out.push_back(static_cast<char>((value >> 8) & 0xff));
    out.push_back(static_cast<char>(value & 0xff));
}

std::string encode_matrix_packet(const Matrix& matrix) {
    std::string out;
    out.reserve(static_cast<std::size_t>(matrix.get_rows()) * matrix.get_cols() * 16 + 16);
    out.append("MCTSBAS1", 8);
    append_u32_be(out, static_cast<std::uint32_t>(matrix.get_rows()));
    append_u32_be(out, static_cast<std::uint32_t>(matrix.get_cols()));

    for (int i = 0; i < matrix.get_rows(); ++i) {
        for (int j = 0; j < matrix.get_cols(); ++j) {
            mpz_srcptr value = matrix[i][j].get_data();
            const int sign = mpz_sgn(value);
            out.push_back(static_cast<char>(sign < 0 ? 0xff : sign > 0 ? 0x01 : 0x00));
            if (sign == 0) {
                append_u32_be(out, 0);
                continue;
            }
            const std::size_t count_max = (mpz_sizeinbase(value, 2) + 7) / 8;
            std::string mag(count_max, '\0');
            std::size_t written = 0;
            mpz_export(mag.data(), &written, 1, 1, 1, 0, value);
            mag.resize(written);
            append_u32_be(out, static_cast<std::uint32_t>(written));
            out.append(mag);
        }
    }
    return out;
}

long double log_abs_mpz(const mpz_class& value) {
    if (value == 0) return -std::numeric_limits<long double>::infinity();
    long exponent = 0;
    const double mantissa = mpz_get_d_2exp(&exponent, value.get_mpz_t());
    return std::log(std::fabs(static_cast<long double>(mantissa))) +
           static_cast<long double>(exponent) * std::log(2.0L);
}

mpz_class exact_square_determinant(const Matrix& matrix) {
    const int n = matrix.get_rows();
    if (n != matrix.get_cols()) {
        throw std::runtime_error("quality gate currently requires a square basis");
    }
    std::vector<std::vector<mpz_class>> a(n, std::vector<mpz_class>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            mpz_set(a[i][j].get_mpz_t(), matrix[i][j].get_data());
        }
    }
    if (n == 0) return 1;
    mpz_class prev = 1;
    int sign = 1;
    for (int k = 0; k < n - 1; ++k) {
        int pivot = k;
        while (pivot < n && a[pivot][k] == 0) ++pivot;
        if (pivot == n) return 0;
        if (pivot != k) {
            std::swap(a[pivot], a[k]);
            sign = -sign;
        }
        const mpz_class p = a[k][k];
        for (int i = k + 1; i < n; ++i) {
            for (int j = k + 1; j < n; ++j) {
                mpz_class numerator = a[i][j] * p - a[i][k] * a[k][j];
                if (k > 0) mpz_divexact(numerator.get_mpz_t(), numerator.get_mpz_t(), prev.get_mpz_t());
                a[i][j] = std::move(numerator);
            }
        }
        prev = p;
    }
    mpz_class det = a[n - 1][n - 1];
    if (sign < 0) det = -det;
    if (det < 0) det = -det;
    return det;
}

py::bytes export_matrix_packet_api(int64_t id) {
    std::lock_guard<std::mutex> lock(g_matrix_pool_mutex);
    const auto iterator = g_matrix_pool.find(id);
    if (iterator == g_matrix_pool.end()) throw std::runtime_error("invalid matrix id");
    return py::bytes(encode_matrix_packet(iterator->second));
}

py::dict quality_metrics_api(int64_t id) {
    std::lock_guard<std::mutex> lock(g_matrix_pool_mutex);
    const auto iterator = g_matrix_pool.find(id);
    if (iterator == g_matrix_pool.end()) throw std::runtime_error("invalid matrix id");
    const Matrix& basis = iterator->second;
    const int n = basis.get_rows();
    if (n <= 0 || basis.get_cols() != n) {
        throw std::runtime_error("quality metrics require a non-empty square basis");
    }

    mpz_class b1_sq = 0;
    mpz_class term;
    for (int j = 0; j < basis.get_cols(); ++j) {
        mpz_class x;
        mpz_set(x.get_mpz_t(), basis[0][j].get_data());
        term = x * x;
        b1_sq += term;
    }
    const mpz_class det = exact_square_determinant(basis);
    if (det == 0 || b1_sq == 0) throw std::runtime_error("singular basis or zero first vector");

    const long double log_b1 = 0.5L * log_abs_mpz(b1_sq);
    const long double log_det = log_abs_mpz(det);
    const long double log_gh =
        (std::lgamma(static_cast<long double>(n) / 2.0L + 1.0L) -
         static_cast<long double>(n) / 2.0L * std::log(std::acos(-1.0L)) +
         log_det) / static_cast<long double>(n);
    const long double ratio = std::exp(log_b1 - log_gh);

    py::dict out;
    out["dimension"] = n;
    out["b1_over_gh"] = static_cast<double>(ratio);
    out["log_b1"] = static_cast<double>(log_b1);
    out["log_gh"] = static_cast<double>(log_gh);
    return out;
}

void shutdown_backend_api() {
    py::gil_scoped_release release;
    lattice_backend::shutdown_local_sieve_engine();
}

}  // namespace

PYBIND11_MODULE(heavy_backend, module) {
    module.doc() =
        "Heavy exact MPZ lattice backend for LLL/BKZ/BGJ preparation";

    module.def("create_matrix", &create_matrix, py::arg("matrix_str"));
    module.def("create_matrix_lll", &create_matrix_lll, py::arg("matrix_str"));
    module.def("reduce_extreme", &reduce_extreme_api,
               py::arg("matrix_id"), py::arg("pos"), py::arg("beta"),
               py::arg("bool_sieve"));
    module.def("reduce_bkz2_global", &reduce_bkz2_global_api,
               py::arg("matrix_id"), py::arg("beta"), py::arg("loops") = 1);
    module.def("extract_block", &extract_block_api,
               py::arg("matrix_id"), py::arg("pos"), py::arg("beta"),
               "Serialize one exact MPZ row block owned by the current process.");
    module.def("sieve_reduce_serialized", &sieve_reduce_serialized_api,
               py::arg("matrix_str"), py::arg("beta"),
               py::arg("max_candidates") = 1,
               py::arg("max_rounds") = 1,
               py::arg("max_pairs") = 0,
               py::arg("time_budget_s") = 0.0,
               py::arg("memory_budget_mb") = 0,
               py::arg("min_b1_rel_improvement") = 0.0,
               py::arg("min_logpot_improvement") = 0.0,
               py::arg("free_dim") = -1,
               py::arg("free_dim_cap") = 0,
               "Persistent-worker BGJ entry point. Exactly one complete BGJ invocation is allowed per call.");
    module.def("apply_external_block", &apply_external_block_api,
               py::arg("matrix_id"), py::arg("pos"), py::arg("matrix_str"),
               py::arg("post_bkz_loops") = 0,
               py::arg("run_full_lll_after") = false,
               "Validate an exact unimodular external block and commit it through the full-basis non-worsening gate.");
    module.def("full_lll", &full_lll_api, py::arg("matrix_id"),
               "Run exact full-basis LLL with rollback on transaction-metric worsening.");
    module.def("reduce_sieve_block", &reduce_sieve_block_api,
               py::arg("matrix_id"), py::arg("pos"), py::arg("beta"));
    module.def("reduce_serialized_block", &reduce_serialized_block_api,
               py::arg("matrix_str"), py::arg("bool_sieve") = true,
               "Worker-safe exact block IPC: returns a serialized MPZ block and explicit result semantics.");
    module.def("reduce", &reduce_compat,
               py::arg("matrix_id"), py::arg("method"), py::arg("param"),
               py::arg("pos"));
    module.def("evaluate_matrix", &evaluate_matrix, py::arg("matrix_id"));
    module.def("dump_matrix", &dump_matrix_api, py::arg("matrix_id"));
    module.def("free_matrix", &free_matrix, py::arg("matrix_id"));
    module.def("clone_matrix", &clone_matrix, py::arg("matrix_id"));
    module.def("minimum_action_beta", &minimum_action_beta_api);
    module.def("adaptive_sieve_threshold", &adaptive_sieve_threshold_api);
    module.def("action_uses_gpu", &action_uses_gpu_api,
               py::arg("matrix_id"), py::arg("pos"), py::arg("beta"));
    module.def("export_matrix_packet", &export_matrix_packet_api, py::arg("matrix_id"),
               "Export the exact MPZ basis through the in-memory MCTSBAS1 packet protocol.");
    module.def("quality_metrics", &quality_metrics_api, py::arg("matrix_id"),
               "Return exact-determinant b1/GH quality metrics for a square basis.");
    module.def("shutdown_backend", &shutdown_backend_api,
               "Release the global pinned sieve allocator when a persistent worker exits.");
#ifdef USE_CUDA
    module.def("cuda_available", []() { return cuda_is_available(); });
#else
    module.def("cuda_available", []() { return false; });
#endif
}
