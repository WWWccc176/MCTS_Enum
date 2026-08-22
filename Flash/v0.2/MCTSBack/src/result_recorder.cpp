#include "mcts/result_recorder.hpp"

#include "mcts/basis.hpp"

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

void write_i64_vector(std::ostream& out, const std::vector<std::int64_t>& values) {
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

double norm_over_gh(const mpz_class& squared_norm, long double log_gh) {
    if (squared_norm <= 0) return 0.0;
    return static_cast<double>(std::exp(
        0.5L * Basis::log_positive_mpz(squared_norm) - log_gh));
}

}  // namespace

void ResultRecorder::add_phase(PhaseSnapshot snapshot) {
    phases_.push_back(std::move(snapshot));
}

void ResultRecorder::write(
    const std::string& result_root,
    const std::string& version,
    const std::string& run_id) const {
    const fs::path root = fs::path(result_root) / version / run_id;
    fs::remove_all(root);
    fs::create_directories(root);

    std::uint64_t total_tree = 0;
    std::uint64_t total_enum = 0;
    std::uint64_t total_work = 0;
    std::uint64_t total_rollouts = 0;
    std::uint64_t total_candidates = 0;
    const PhaseSnapshot* global_best = nullptr;
    mpz_class global_best_sq = 0;
    std::uint64_t work_before = 0;
    std::uint64_t global_best_at_total_work = 0;
    std::uint32_t global_best_phase = 0;

    for (const auto& phase : phases_) {
        total_tree += phase.tree_nodes;
        total_enum += phase.enumeration_nodes;
        total_work += phase.work_nodes;
        total_rollouts += phase.rollout_jobs;
        total_candidates += phase.exact_candidates;
        if (!global_best || phase.best_squared_norm < global_best_sq) {
            global_best = &phase;
            global_best_sq = phase.best_squared_norm;
            global_best_phase = phase.phase_index;
            global_best_at_total_work = work_before + phase.best_found_at_work_node_count;
        }
        work_before += phase.work_nodes;
    }

    {
        std::ostringstream out;
        out << std::setprecision(12)
            << "version=" << version << '\n'
            << "algorithm=hybrid-mcts-fplll-se\n"
            << "variant=flash\n"
            << "run_id=" << run_id << '\n'
            << "phase_count=" << phases_.size() << '\n'
            << "total_tree_nodes_all_phases=" << total_tree << '\n'
            << "total_enumeration_nodes_all_phases=" << total_enum << '\n'
            << "total_work_nodes_all_phases=" << total_work << '\n'
            << "total_rollout_jobs_all_phases=" << total_rollouts << '\n'
            << "total_exact_candidates_all_phases=" << total_candidates << '\n';

        if (!phases_.empty() && global_best) {
            const auto& first = phases_.front();
            out << "initial_search_radius_over_gh=" << first.initial_radius_over_gh << '\n'
                << "final_best_over_gh=" << norm_over_gh(global_best_sq, first.log_gh) << '\n'
                << "final_best_squared_norm=" << global_best_sq.get_str() << '\n'
                << "final_best_found_in_phase=" << global_best_phase << '\n'
                << "final_best_found_at_total_work_node_count="
                << global_best_at_total_work << '\n';
        }

        if (has_search_config_) {
            out << "work_budget_per_phase="
                << (search_config_.unlimited_nodes ? -1LL
                    : static_cast<long long>(search_config_.node_budget)) << '\n'
                << "search_threads=" << search_config_.search_threads << '\n'
                << "rollout_dimensions=" << search_config_.rollout_dimensions << '\n'
                << "rollout_solutions=" << search_config_.rollout_solutions << '\n'
                << "w_m=" << search_config_.w_m << '\n'
                << "lambda_puct=" << search_config_.lambda_puct << '\n'
                << "c_pw=" << search_config_.cpw << '\n'
                << "d_pw=" << search_config_.dpw << '\n'
                << "lll_delta=" << search_config_.lll_delta << '\n'
                << "refresh_potential_rel_tolerance="
                << search_config_.refresh_potential_rel_tolerance << '\n';
        }

        for (const auto& phase : phases_) {
            out << "\n[phase " << phase.phase_index << "]\n"
                << "tree_nodes=" << phase.tree_nodes << '\n'
                << "enumeration_nodes=" << phase.enumeration_nodes << '\n'
                << "work_nodes=" << phase.work_nodes << '\n'
                << "rollout_jobs=" << phase.rollout_jobs << '\n'
                << "exact_candidates=" << phase.exact_candidates << '\n'
                << "best_found_at_work_node_count=" << phase.best_found_at_work_node_count << '\n'
                << "initial_radius_over_gh=" << phase.initial_radius_over_gh << '\n'
                << "best_over_gh=" << norm_over_gh(phase.best_squared_norm, phase.log_gh) << '\n'
                << "best_squared_norm=" << phase.best_squared_norm.get_str() << '\n'
                << "refresh_after=" << (phase.basis_refresh_after ? 1 : 0) << '\n'
                << "refresh_changed=" << (phase.basis_refresh_changed ? 1 : 0) << '\n'
                << "refresh_accepted=" << (phase.basis_refresh_accepted ? 1 : 0) << '\n'
                << "refresh_potential_before=" << phase.refresh_potential_before << '\n'
                << "refresh_potential_after=" << phase.refresh_potential_after << '\n'
                << "stop_reason="
                << (phase.tree_exhausted ? "tree_exhausted"
                    : phase.budget_reached ? "work_budget"
                    : phase.basis_refresh_after ? "basis_refresh" : "finalized")
                << '\n';
        }
        write_text(root / "summary.txt", out.str());
    }

    {
        std::ostringstream out;
        for (const auto& phase : phases_) {
            out << "[phase " << phase.phase_index << " initial]\n" << phase.initial_basis_text;
            if (!phase.initial_basis_text.empty() && phase.initial_basis_text.back() != '\n') out << '\n';
            if (phase.basis_refresh_after) {
                out << "[phase " << phase.phase_index << " refreshed accepted="
                    << (phase.basis_refresh_accepted ? 1 : 0) << "]\n"
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
                << "squared_norm=" << phase.best_squared_norm.get_str() << '\n'
                << "coefficients=";
            write_i64_vector(out, phase.best_coefficients);
            out << "\nvector=";
            write_mpz_vector(out, phase.best_vector);
            out << "\n\n";
        }
        write_text(root / "best.txt", out.str());
    }

    {
        std::ostringstream out;
        out << "phase\tdepth\tk\tnode\tparent\tz_k\ttree_nodes_at_depth\tparent_rho\trho\tbound_r2\tcenter\tdelta\tlegal_lo\tlegal_hi\tN\tQ\tM\n";
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
                    << static_cast<double>(row.center) << '\t'
                    << static_cast<double>(row.delta) << '\t'
                    << row.legal_lo << '\t' << row.legal_hi << '\t'
                    << row.edge_visits << '\t' << row.q << '\t' << row.m << '\n';
            }
        }
        write_text(root / "path.txt", out.str());
    }

    {
        std::ostringstream out;
        out << "phase\tdepth\tcoefficient\ttree_nodes\tenumeration_nodes\n";
        for (const auto& phase : phases_) {
            const std::size_t n = phase.nodes_per_depth.empty() ? 0 : phase.nodes_per_depth.size() - 1;
            for (std::size_t depth = 0; depth < phase.nodes_per_depth.size(); ++depth) {
                const long k = depth == 0 ? -1L : static_cast<long>(n - depth);
                std::uint64_t enum_nodes = 0;
                if (k >= 0 && static_cast<std::size_t>(k) < phase.enumeration_nodes_per_level.size()) {
                    enum_nodes = phase.enumeration_nodes_per_level[static_cast<std::size_t>(k)];
                }
                out << phase.phase_index << '\t' << depth << '\t' << k << '\t'
                    << phase.nodes_per_depth[depth] << '\t' << enum_nodes << '\n';
            }
        }
        write_text(root / "depth.txt", out.str());
    }
}

}  // namespace mcts_enum
