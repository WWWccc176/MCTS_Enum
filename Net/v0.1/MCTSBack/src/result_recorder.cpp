#include "mcts/result_recorder.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace mcts_enum {
namespace fs = std::filesystem;
namespace {

void write_text(const fs::path& path, const std::string& text) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot open result file: " + path.string());
    out << text;
}

template <class T>
void write_vector(std::ostream& out, const std::vector<T>& values) {
    out << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) out << ' ';
        out << values[i];
    }
    out << ']';
}

void write_mpz_vector(std::ostream& out, const std::vector<mpz_class>& values) {
    out << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) out << ' ';
        out << values[i].get_str();
    }
    out << ']';
}

double best_over_gh(const PhaseSnapshot& phase, int dimension) {
    if (dimension <= 0) return phase.initial_b1_over_gh;
    return phase.initial_b1_over_gh * std::exp(-phase.best_score / static_cast<double>(dimension));
}

}  // namespace

void ResultRecorder::add_phase(PhaseSnapshot snapshot) {
    phases_.push_back(std::move(snapshot));
}

void ResultRecorder::add_nn_metric(const NnMetricRecord& metric) {
    nn_metrics_.push_back(metric);
}

void ResultRecorder::write(
    const std::string& result_root,
    const std::string& version,
    const std::string& run_id) const {
    const fs::path root = fs::path(result_root) / version / run_id;
    fs::remove_all(root);
    fs::create_directories(root);

    std::uint64_t total_nodes = 0;
    for (const auto& phase : phases_) total_nodes += phase.total_nodes;

    std::uint32_t final_best_found_phase = 0;
    std::uint64_t final_best_found_phase_node = 0;
    std::uint64_t final_best_found_total_node = 0;
    if (!phases_.empty()) {
        const mpz_class& final_norm = phases_.back().best_squared_norm;
        std::uint64_t nodes_before = 0;
        for (const auto& phase : phases_) {
            if (phase.best_squared_norm == final_norm) {
                final_best_found_phase = phase.phase_index;
                final_best_found_phase_node = phase.best_found_at_node_count;
                final_best_found_total_node = nodes_before + phase.best_found_at_node_count;
                break;
            }
            nodes_before += phase.total_nodes;
        }
    }

    {
        std::ostringstream out;
        out << std::setprecision(12)
            << "version=" << version << '\n'
            << "algorithm=v0.1\n"
            << "variant=" << (nn_enabled_ ? "net" : "flash") << '\n'
            << "run_id=" << run_id << '\n'
            << "nn_enabled=" << (nn_enabled_ ? 1 : 0) << '\n'
            << "phase_count=" << phases_.size() << '\n'
            << "total_nodes_all_phases=" << total_nodes << '\n';
        if (!phases_.empty()) {
            const auto& first = phases_.front();
            const auto& last = phases_.back();
            const int dimension = static_cast<int>(last.best_coefficients.size());
            const double global_score =
                (dimension > 0 && first.initial_radius_squared > 0 && last.best_squared_norm > 0)
                ? 0.5 * static_cast<double>(dimension) * static_cast<double>(
                    Basis::log_positive_mpz(first.initial_radius_squared) -
                    Basis::log_positive_mpz(last.best_squared_norm))
                : 0.0;
            out << "initial_b1_over_gh=" << first.initial_b1_over_gh << '\n'
                << "final_best_over_gh=" << best_over_gh(last, dimension) << '\n'
                << "final_best_score_global=" << global_score << '\n'
                << "final_phase_best_score=" << last.best_score << '\n'
                << "final_best_squared_norm=" << last.best_squared_norm.get_str() << '\n'
                << "final_best_found_phase=" << final_best_found_phase << '\n'
                << "final_best_found_at_phase_node_count=" << final_best_found_phase_node << '\n'
                << "final_best_found_at_total_node_count=" << final_best_found_total_node << '\n';
        }
        if (has_search_config_) {
            out << "node_budget_per_phase="
                << (search_config_.unlimited_nodes ? -1LL
                                                   : static_cast<long long>(search_config_.node_budget)) << '\n'
                << "search_threads=" << search_config_.search_threads << '\n'
                << "nn_refresh_nodes=" << search_config_.refresh_interval << '\n'
                << "radius_refresh_drops=" << search_config_.radius_global_update_interval << '\n'
                << "w_m=" << search_config_.w_m << '\n'
                << "lambda_puct=" << search_config_.lambda_puct << '\n'
                << "c_pw=" << search_config_.cpw << '\n'
                << "d_pw=" << search_config_.dpw << '\n'
                << "policy_mix=" << search_config_.policy_mix << '\n'
                << "visit_temperature=" << search_config_.visit_temperature << '\n'
                << "quality_gate=" << search_config_.quality_gate << '\n';
        }
        for (const auto& phase : phases_) {
            const int dimension = static_cast<int>(phase.best_coefficients.size());
            out << "\n[phase " << phase.phase_index << "]\n"
                << "total_nodes=" << phase.total_nodes << '\n'
                << "best_found_at_node_count=" << phase.best_found_at_node_count << '\n'
                << "initial_b1_over_gh=" << phase.initial_b1_over_gh << '\n'
                << "best_over_gh=" << best_over_gh(phase, dimension) << '\n'
                << "phase_best_score=" << phase.best_score << '\n'
                << "best_squared_norm=" << phase.best_squared_norm.get_str() << '\n'
                << "initial_radius_squared=" << phase.initial_radius_squared.get_str() << '\n'
                << "gso_scale_exp2=" << phase.gso_scale_exp2 << '\n'
                << "refresh_after=" << (phase.basis_refresh_after ? 1 : 0) << '\n'
                << "refresh_changed=" << (phase.basis_refresh_changed ? 1 : 0) << '\n'
                << "stop_reason="
                << (phase.tree_exhausted ? "tree_exhausted"
                    : phase.budget_reached ? "node_budget"
                    : phase.basis_refresh_after ? "basis_refresh" : "finalized") << '\n';
        }
        write_text(root / "summary.txt", out.str());
    }

    {
        std::ostringstream out;
        for (const auto& phase : phases_) {
            out << "[phase " << phase.phase_index << " initial]\n"
                << phase.initial_basis_text;
            if (!phase.initial_basis_text.empty() && phase.initial_basis_text.back() != '\n') out << '\n';
            if (phase.basis_refresh_after) {
                out << "[phase " << phase.phase_index << " refreshed]\n"
                    << phase.refreshed_basis_text;
                if (!phase.refreshed_basis_text.empty() && phase.refreshed_basis_text.back() != '\n') out << '\n';
            }
        }
        write_text(root / "basis.txt", out.str());
    }

    {
        std::ostringstream out;
        out << std::setprecision(16);
        for (const auto& phase : phases_) {
            out << "[phase " << phase.phase_index << "]\n"
                << "score=" << phase.best_score << '\n'
                << "squared_norm=" << phase.best_squared_norm.get_str() << '\n'
                << "coefficients=";
            write_vector(out, phase.best_coefficients);
            out << "\nvector=";
            write_mpz_vector(out, phase.best_vector);
            out << "\n\n";
        }
        write_text(root / "best.txt", out.str());
    }

    {
        std::ostringstream out;
        out << "phase\tdepth\tk\tnode\tparent\tz_k\tnodes_at_depth\tparent_rho\trho\tbound_r2\tremaining\tcenter\tdelta\tlegal_lo\tlegal_hi\toffset\tP\tPtheta\tN\tQ\tM\n";
        out << std::setprecision(17);
        for (const auto& phase : phases_) {
            for (const auto& row : phase.best_path) {
                out << phase.phase_index << '\t' << row.depth << '\t'
                    << row.coefficient_index << '\t' << row.node_id << '\t'
                    << row.parent_id << '\t' << row.action << '\t'
                    << row.nodes_at_depth << '\t'
                    << static_cast<double>(row.parent_rho) << '\t'
                    << static_cast<double>(row.rho) << '\t'
                    << static_cast<double>(row.radius_sq) << '\t'
                    << static_cast<double>(row.remaining_sq) << '\t'
                    << static_cast<double>(row.center) << '\t'
                    << static_cast<double>(row.delta) << '\t'
                    << row.legal_lo << '\t' << row.legal_hi << '\t'
                    << row.normalized_offset << '\t' << row.prior << '\t'
                    << row.policy_prior << '\t' << row.edge_visits << '\t'
                    << row.q << '\t' << row.m << '\n';
            }
        }
        write_text(root / "path.txt", out.str());
    }

    {
        std::ostringstream out;
        out << "phase\tdepth\tcoefficient\tnodes\n";
        for (const auto& phase : phases_) {
            const std::size_t n = phase.nodes_per_depth.empty() ? 0 : phase.nodes_per_depth.size() - 1;
            for (std::size_t depth = 0; depth < phase.nodes_per_depth.size(); ++depth) {
                const long coefficient = depth == 0 ? -1L : static_cast<long>(n - depth);
                out << phase.phase_index << '\t' << depth << '\t'
                    << coefficient << '\t' << phase.nodes_per_depth[depth] << '\n';
            }
        }
        write_text(root / "depth.txt", out.str());
    }

    {
        std::ostringstream out;
        out << "enabled=" << (nn_enabled_ ? 1 : 0) << '\n';
        out << "update\tnodes\tpolicy_loss\tvalue_loss\ttotal_loss\tlr\n";
        out << std::setprecision(10);
        for (const auto& metric : nn_metrics_) {
            out << metric.update_index << '\t' << metric.node_count << '\t'
                << metric.policy_loss << '\t' << metric.value_loss << '\t'
                << metric.total_loss << '\t' << metric.learning_rate << '\n';
        }
        write_text(root / "train.txt", out.str());
    }
}

}  // namespace mcts_enum
