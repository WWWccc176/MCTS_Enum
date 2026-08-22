#include "mcts/search_engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <exception>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace mcts_enum {
namespace {

std::uint64_t legal_count(const GeometryInfo& geometry) {
    if (geometry.pruned || geometry.hi < geometry.lo) return 0;
    const __int128 width = static_cast<__int128>(geometry.hi) -
                           static_cast<__int128>(geometry.lo) + 1;
    if (width <= 0) return 0;
    if (width > static_cast<__int128>(std::numeric_limits<std::uint64_t>::max())) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    return static_cast<std::uint64_t>(width);
}

bool action_in_geometry(const GeometryInfo& geometry, std::int64_t action) {
    return !geometry.pruned && action >= geometry.lo && action <= geometry.hi;
}

}  // namespace

SearchEngine::SearchEngine(Basis basis, SearchConfig config, bool nn_enabled)
    : basis_(std::move(basis)),
      config_(config),
      nn_enabled_(nn_enabled),
      geometry_(gso_, config_) {
    if (nn_enabled_) {
        throw std::runtime_error("Flash v0.2 is search-only; use Net for neural inference");
    }
    if (!config_.unlimited_nodes && config_.node_budget == 0) {
        throw std::runtime_error("node_budget must be positive unless unlimited_nodes is enabled");
    }
    if (config_.search_threads == 0) throw std::runtime_error("search_threads must be positive");
    if (config_.rollout_dimensions == 0) throw std::runtime_error("rollout_dimensions must be positive");
    if (config_.rollout_solutions == 0) throw std::runtime_error("rollout_solutions must be positive");
    if (!(config_.w_m >= 0.0 && config_.w_m <= 1.0)) throw std::runtime_error("w_m must be in [0,1]");
    if (!(config_.dpw > 0.0 && config_.dpw < 1.0)) throw std::runtime_error("dpw must be in (0,1)");
    if (!(config_.cpw > 0.0)) throw std::runtime_error("cpw must be positive");
    if (!(config_.lambda_puct >= 0.0)) throw std::runtime_error("lambda_puct must be non-negative");
    if (!(config_.lll_delta > 0.5 && config_.lll_delta < 1.0)) throw std::runtime_error("lll_delta must be in (0.5,1)");

    recorder_.set_search_config(config_);
    initialize_episode();
}

std::pair<int, mpz_class> SearchEngine::shortest_basis_row() const {
    int best_index = -1;
    mpz_class best_sq = 0;
    for (int i = 0; i < basis_.dimension(); ++i) {
        mpz_class row_sq = 0;
        for (int j = 0; j < basis_.columns(); ++j) {
            mpz_class x;
            mpz_set(x.get_mpz_t(), basis_.matrix()[i][j].get_data());
            row_sq += x * x;
        }
        if (row_sq <= 0) continue;
        if (best_index < 0 || row_sq < best_sq) {
            best_index = i;
            best_sq = row_sq;
        }
    }
    if (best_index < 0) throw std::runtime_error("basis contains no non-zero row");
    return {best_index, best_sq};
}

void SearchEngine::initialize_episode() {
    gso_.recompute(basis_);
    const auto [shortest_index, shortest_sq] = shortest_basis_row();

    phase_initial_radius_sq_ = shortest_sq;
    phase_best_sq_ = shortest_sq;
    phase_best_z_.assign(static_cast<std::size_t>(basis_.dimension()), 0);
    phase_best_z_[static_cast<std::size_t>(shortest_index)] = 1;
    phase_best_vector_ = basis_.exact_vector(phase_best_z_);
    phase_best_found_at_work_ = 0;
    phase_best_path_.clear();

    if (overall_best_sq_ <= 0 || phase_best_sq_ < overall_best_sq_) {
        overall_best_sq_ = phase_best_sq_;
        overall_best_vector_ = phase_best_vector_;
    }

    refresh_candidate_sq_ = 0;
    refresh_candidate_z_.clear();
    refresh_candidate_found_at_work_ = 0;

    input_quality_ratio_ = static_cast<double>(std::exp(
        0.5L * Basis::log_positive_mpz(phase_initial_radius_sq_) - gso_.log_gh()));

    const std::size_t reserve_nodes = config_.unlimited_nodes
        ? 4096
        : static_cast<std::size_t>(std::min<std::uint64_t>(config_.node_budget, 1'000'000ULL));
    tree_.reset(basis_.dimension(), reserve_nodes);
    phase_initial_basis_ = basis_.to_text();
    phase_snapshot_committed_ = false;
    radius_epoch_ = 0;
    radius_drops_ = 0;
    enumeration_nodes_.store(0, std::memory_order_relaxed);
    rollout_jobs_.store(0, std::memory_order_relaxed);
    exact_candidates_.store(0, std::memory_order_relaxed);
    active_rollouts_.store(0, std::memory_order_relaxed);
    enumeration_nodes_per_level_.assign(static_cast<std::size_t>(basis_.dimension()), 0);

    Node& root = tree_.node(tree_.root_id());
    refresh_geometry_locked(root);
}

std::uint64_t SearchEngine::node_count() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return static_cast<std::uint64_t>(tree_.size());
}

std::uint64_t SearchEngine::work_node_count() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return work_nodes_locked();
}

std::uint64_t SearchEngine::work_nodes_locked() const noexcept {
    return static_cast<std::uint64_t>(tree_.size()) +
           enumeration_nodes_.load(std::memory_order_relaxed);
}

bool SearchEngine::budget_reached_locked() const noexcept {
    return !config_.unlimited_nodes && work_nodes_locked() >= config_.node_budget;
}

bool SearchEngine::finished() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return active_rollouts_.load(std::memory_order_relaxed) == 0 &&
           (budget_reached_locked() || tree_.node(tree_.root_id()).closed);
}

long double SearchEngine::radius_sq_scaled_locked() const {
    return Basis::scaled_positive_mpz(phase_best_sq_, gso_.square_scale_exp2());
}

void SearchEngine::refresh_geometry_locked(Node& node) {
    if (node.closed || node.next_k < 0) return;
    if (node.geometry_ready && node.geometry_epoch == radius_epoch_) return;

    if (!node.geometry_ready) {
        node.geometry = geometry_.legal_interval(
            node.next_k,
            node.rho,
            radius_sq_scaled_locked(),
            node.prefix_high_to_low);
        node.geometry_ready = true;
        if (!node.geometry.pruned) {
            const long double f = std::floor(node.geometry.center);
            if (f < static_cast<long double>(std::numeric_limits<std::int64_t>::min()) ||
                f > static_cast<long double>(std::numeric_limits<std::int64_t>::max())) {
                throw std::runtime_error("SE center exceeds int64 range");
            }
            node.se_left = static_cast<std::int64_t>(f);
            node.se_left_done = node.se_left < node.geometry.lo || node.se_left > node.geometry.hi;
            if (node.se_left == std::numeric_limits<std::int64_t>::max()) {
                node.se_right = node.se_left;
                node.se_right_done = true;
            } else {
                node.se_right = node.se_left + 1;
                node.se_right_done =
                    node.se_right < node.geometry.lo || node.se_right > node.geometry.hi;
            }
            node.se_initialized = true;
            node.se_exhausted = node.se_left_done && node.se_right_done;
        }
    } else {
        node.geometry = geometry_.legal_interval_from_center(
            node.next_k,
            node.rho,
            radius_sq_scaled_locked(),
            node.geometry.center);
    }
    node.geometry_epoch = radius_epoch_;

    if (node.geometry.pruned) {
        if (node.in_flight == 0) node.closed = true;
        for (auto& edge : node.edges) edge.active = false;
        return;
    }

    for (auto& edge : node.edges) {
        edge.active = action_in_geometry(node.geometry, edge.action);
    }
}

bool SearchEngine::generate_next_se_edge_locked(Node& node) {
    refresh_geometry_locked(node);
    if (node.closed || node.geometry.pruned || node.se_exhausted) return false;

    if (!node.se_left_done &&
        (node.se_left < node.geometry.lo || node.se_left > node.geometry.hi)) {
        node.se_left_done = true;
    }
    if (!node.se_right_done &&
        (node.se_right < node.geometry.lo || node.se_right > node.geometry.hi)) {
        node.se_right_done = true;
    }
    if (node.se_left_done && node.se_right_done) {
        node.se_exhausted = true;
        return false;
    }

    const bool left_valid = !node.se_left_done;
    const bool right_valid = !node.se_right_done;
    bool choose_left = left_valid;
    if (left_valid && right_valid) {
        const long double dl = std::fabs(
            static_cast<long double>(node.se_left) - node.geometry.center);
        const long double dr = std::fabs(
            static_cast<long double>(node.se_right) - node.geometry.center);
        choose_left = dl < dr || (dl == dr && node.se_left < node.se_right);
    } else if (!left_valid) {
        choose_left = false;
    }

    const std::int64_t action = choose_left ? node.se_left : node.se_right;
    if (choose_left) {
        if (node.se_left == std::numeric_limits<std::int64_t>::min() ||
            node.se_left <= node.geometry.lo) {
            node.se_left_done = true;
        } else {
            --node.se_left;
            if (node.se_left < node.geometry.lo) node.se_left_done = true;
        }
    } else {
        if (node.se_right == std::numeric_limits<std::int64_t>::max() ||
            node.se_right >= node.geometry.hi) {
            node.se_right_done = true;
        } else {
            ++node.se_right;
            if (node.se_right > node.geometry.hi) node.se_right_done = true;
        }
    }
    node.se_exhausted = node.se_left_done && node.se_right_done;

    const long double norm_abs = node.geometry.delta > 0.0L
        ? std::fabs(
              (static_cast<long double>(action) - node.geometry.center) /
              node.geometry.delta)
        : 0.0L;
    EdgeStats edge;
    edge.action = action;
    edge.prior_weight = static_cast<double>(
        std::exp(-2.0L * norm_abs * norm_abs) + 1.0e-12L);
    edge.active = true;
    node.edges.push_back(edge);
    return true;
}

void SearchEngine::ensure_widened_locked(Node& node) {
    refresh_geometry_locked(node);
    if (node.closed || node.geometry.pruned) return;
    const std::uint64_t total_legal = legal_count(node.geometry);
    if (total_legal == 0) {
        if (node.in_flight == 0) node.closed = true;
        return;
    }

    const double visits = static_cast<double>(
        std::max<std::uint64_t>(1, node.visits + node.in_flight + 1));
    std::uint64_t allowed = static_cast<std::uint64_t>(std::floor(
        config_.cpw * std::pow(visits, config_.dpw)));
    allowed = std::max<std::uint64_t>(1, std::min(total_legal, allowed));

    auto active_generated = [&]() {
        return static_cast<std::uint64_t>(std::count_if(
            node.edges.begin(), node.edges.end(),
            [](const EdgeStats& edge) { return edge.active; }));
    };

    while (active_generated() < allowed) {
        if (!generate_next_se_edge_locked(node)) break;
    }
}

std::size_t SearchEngine::choose_edge_locked(const Node& node) const {
    double total_prior = 0.0;
    for (const auto& edge : node.edges) {
        if (!edge.active) continue;
        if (edge.expanded && edge.child_id >= 0) {
            const Node& child = tree_.node(edge.child_id);
            if (child.closed || child.pending_rollout) continue;
        }
        total_prior += edge.prior_weight;
    }
    if (!(total_prior > 0.0)) return node.edges.size();

    const double parent_visits = static_cast<double>(
        std::max<std::uint64_t>(1, node.visits + node.in_flight));
    const double sqrt_parent = std::sqrt(parent_visits);
    double best_score = -std::numeric_limits<double>::infinity();
    std::size_t best = node.edges.size();

    for (std::size_t i = 0; i < node.edges.size(); ++i) {
        const EdgeStats& edge = node.edges[i];
        if (!edge.active) continue;
        if (edge.expanded && edge.child_id >= 0) {
            const Node& child = tree_.node(edge.child_id);
            if (child.closed || child.pending_rollout) continue;
        }

        const double prior = edge.prior_weight / total_prior;
        const double m_effective = edge.has_exact_m ? edge.m : edge.q;
        const double exploitation = (1.0 - config_.w_m) * edge.q + config_.w_m * m_effective;
        const double exploration = config_.lambda_puct * prior * sqrt_parent /
            (1.0 + static_cast<double>(edge.visits + edge.in_flight));
        const double score = exploitation + exploration;
        if (score > best_score) {
            best_score = score;
            best = i;
        }
    }
    return best;
}

bool SearchEngine::is_rollout_frontier(const Node& node) const noexcept {
    return node.next_k < 0 ||
           static_cast<std::uint32_t>(node.next_k + 1) <= config_.rollout_dimensions;
}

void SearchEngine::reserve_path_locked(const RolloutJob& job) {
    for (const auto& [parent_id, edge_index] : job.path) {
        Node& parent = tree_.node(parent_id);
        EdgeStats& edge = parent.edges.at(edge_index);
        ++parent.in_flight;
        ++edge.in_flight;
    }
    Node& leaf = tree_.node(job.leaf_id);
    ++leaf.in_flight;
    leaf.pending_rollout = true;
    active_rollouts_.fetch_add(1, std::memory_order_relaxed);
    rollout_jobs_.fetch_add(1, std::memory_order_relaxed);
}

void SearchEngine::release_path_locked(const RolloutJob& job) {
    for (const auto& [parent_id, edge_index] : job.path) {
        Node& parent = tree_.node(parent_id);
        EdgeStats& edge = parent.edges.at(edge_index);
        if (parent.in_flight > 0) --parent.in_flight;
        if (edge.in_flight > 0) --edge.in_flight;
    }
    Node& leaf = tree_.node(job.leaf_id);
    if (leaf.in_flight > 0) --leaf.in_flight;
    leaf.pending_rollout = false;
    active_rollouts_.fetch_sub(1, std::memory_order_relaxed);
}

bool SearchEngine::reserve_rollout_locked(RolloutJob& job) {
    if (budget_reached_locked() || tree_.node(tree_.root_id()).closed) return false;

    std::vector<PathStep> path;
    std::int64_t current = tree_.root_id();
    const int guard_limit = std::max(8, basis_.dimension() + 4);

    for (int guard = 0; guard < guard_limit; ++guard) {
        Node& node = tree_.node(current);
        refresh_geometry_locked(node);
        if (node.closed) return false;

        if (is_rollout_frontier(node)) {
            if (node.pending_rollout) return false;
            job.leaf_id = current;
            job.path = path;
            job.prefix_high_to_low = node.prefix_high_to_low;
            job.radius_squared = phase_best_sq_;
            job.radius_epoch = radius_epoch_;
            reserve_path_locked(job);
            return true;
        }

        ensure_widened_locked(node);
        if (node.closed) return false;
        std::size_t edge_index = choose_edge_locked(node);
        if (edge_index >= node.edges.size()) {
            // Progressive widening must never make the search non-live. If every
            // currently opened child is closed/in-flight but legal actions remain,
            // open the next Schnorr-Euchner action immediately.
            if (generate_next_se_edge_locked(node)) {
                edge_index = choose_edge_locked(node);
            }
        }
        if (edge_index >= node.edges.size()) {
            if (node_can_close_locked(node)) node.closed = true;
            return false;
        }

        const std::int64_t parent_id = current;
        const std::int64_t action = node.edges[edge_index].action;
        std::int64_t child_id = node.edges[edge_index].child_id;
        if (!node.edges[edge_index].expanded) {
            const long double child_rho = geometry_.next_rho(
                node.next_k, node.rho, node.geometry.center, action);
            child_id = tree_.create_child(parent_id, edge_index, child_rho);
        }
        path.emplace_back(parent_id, edge_index);
        current = child_id;
    }

    throw std::runtime_error("hybrid selection exceeded depth guard");
}

void SearchEngine::backup_locked(
    const RolloutJob& job,
    double value,
    bool has_exact,
    double exact_score) {
    Node& leaf = tree_.node(job.leaf_id);
    ++leaf.visits;
    for (auto it = job.path.rbegin(); it != job.path.rend(); ++it) {
        Node& parent = tree_.node(it->first);
        EdgeStats& edge = parent.edges.at(it->second);
        ++parent.visits;
        ++edge.visits;
        edge.w += value;
        edge.q = edge.w / static_cast<double>(edge.visits);
        if (has_exact && (!edge.has_exact_m || exact_score > edge.m)) {
            edge.m = exact_score;
            edge.has_exact_m = true;
        }
    }
}

bool SearchEngine::node_can_close_locked(const Node& node) const {
    if (node.pending_rollout || node.in_flight > 0) return false;
    if (node.geometry.pruned) return true;
    const std::uint64_t total_legal = legal_count(node.geometry);
    if (total_legal == 0) return true;

    std::uint64_t generated_legal = 0;
    for (const auto& edge : node.edges) {
        if (!edge.active) continue;
        ++generated_legal;
        if (!edge.expanded || edge.child_id < 0 || edge.in_flight > 0) return false;
        if (!tree_.node(edge.child_id).closed) return false;
    }
    return generated_legal == total_legal;
}

void SearchEngine::propagate_closed_locked(const RolloutJob& job) {
    Node& leaf = tree_.node(job.leaf_id);
    leaf.closed = true;
    for (auto it = job.path.rbegin(); it != job.path.rend(); ++it) {
        Node& parent = tree_.node(it->first);
        refresh_geometry_locked(parent);
        if (node_can_close_locked(parent)) {
            parent.closed = true;
        } else {
            break;
        }
    }
}

bool SearchEngine::is_single_basis_vector(const std::vector<std::int64_t>& z) const {
    int nonzero = 0;
    for (const auto value : z) {
        if (value == 0) continue;
        ++nonzero;
        if (value != 1 && value != -1) return false;
        if (nonzero > 1) return false;
    }
    return nonzero == 1;
}

bool SearchEngine::is_zero_coefficients(const std::vector<std::int64_t>& z) const {
    return std::all_of(z.begin(), z.end(), [](std::int64_t x) { return x == 0; });
}

double SearchEngine::phase_score_from_norm(const mpz_class& norm_sq) const {
    if (norm_sq <= 0 || phase_initial_radius_sq_ <= 0) return 0.0;
    const long double score = 0.5L * static_cast<long double>(basis_.dimension()) *
        (Basis::log_positive_mpz(phase_initial_radius_sq_) - Basis::log_positive_mpz(norm_sq));
    return static_cast<double>(score);
}

double SearchEngine::rollout_value(
    const mpz_class& snapshot_radius_sq,
    const std::vector<ExactCandidate>& exact_candidates) const {
    if (exact_candidates.empty()) return -1.0;
    const auto best_it = std::min_element(
        exact_candidates.begin(), exact_candidates.end(),
        [](const ExactCandidate& a, const ExactCandidate& b) { return a.norm_sq < b.norm_sq; });
    if (best_it == exact_candidates.end() || best_it->norm_sq <= 0) return -1.0;
    const long double log_improvement = 0.5L *
        (Basis::log_positive_mpz(snapshot_radius_sq) - Basis::log_positive_mpz(best_it->norm_sq));
    return static_cast<double>(std::tanh(4.0L * std::max(0.0L, log_improvement)));
}

std::vector<PathRecord> SearchEngine::build_path_records_locked(
    const std::vector<std::int64_t>& z,
    const RolloutJob& job) const {
    std::vector<PathRecord> records;
    if (static_cast<int>(z.size()) != basis_.dimension()) return records;
    records.reserve(z.size());

    const auto depth_counts = tree_.nodes_per_depth();
    const long double snapshot_radius = Basis::scaled_positive_mpz(
        job.radius_squared, gso_.square_scale_exp2());
    std::vector<std::int64_t> prefix;
    prefix.reserve(z.size());
    long double rho = 0.0L;

    for (int depth = 0; depth < basis_.dimension(); ++depth) {
        const int k = basis_.dimension() - 1 - depth;
        const auto geometry = geometry_.legal_interval(k, rho, snapshot_radius, prefix);
        const std::int64_t action = z[static_cast<std::size_t>(k)];
        const long double child_rho = geometry_.next_rho(k, rho, geometry.center, action);

        PathRecord row;
        row.depth = static_cast<std::uint64_t>(depth + 1);
        row.coefficient_index = k;
        row.action = action;
        row.parent_rho = rho;
        row.rho = child_rho;
        row.radius_sq = snapshot_radius;
        row.center = geometry.center;
        row.delta = geometry.delta;
        row.legal_lo = geometry.lo;
        row.legal_hi = geometry.hi;
        if (static_cast<std::size_t>(depth + 1) < depth_counts.size()) {
            row.nodes_at_depth = depth_counts[static_cast<std::size_t>(depth + 1)];
        }

        if (static_cast<std::size_t>(depth) < job.path.size()) {
            const auto [parent_id, edge_index] = job.path[static_cast<std::size_t>(depth)];
            const Node& parent = tree_.node(parent_id);
            const EdgeStats& edge = parent.edges.at(edge_index);
            row.node_id = edge.child_id;
            row.parent_id = parent_id;
            row.edge_visits = edge.visits;
            row.q = edge.q;
            row.m = edge.has_exact_m ? edge.m : 0.0;
        }

        records.push_back(row);
        prefix.push_back(action);
        rho = child_rho;
    }
    return records;
}

std::vector<PathRecord> SearchEngine::best_path_records_locked() const {
    return phase_best_path_;
}

void SearchEngine::commit_rollout_locked(
    const RolloutJob& job,
    const RolloutResult& rollout,
    const std::vector<ExactCandidate>& exact_candidates) {
    const std::uint64_t current_work = work_nodes_locked();
    bool radius_decreased = false;
    bool have_exact = false;
    double best_exact_score = -std::numeric_limits<double>::infinity();

    for (const auto& candidate : exact_candidates) {
        if (candidate.norm_sq <= 0 || is_zero_coefficients(candidate.z)) continue;
        have_exact = true;
        best_exact_score = std::max(best_exact_score, phase_score_from_norm(candidate.norm_sq));

        if (!is_single_basis_vector(candidate.z) &&
            (refresh_candidate_sq_ <= 0 || candidate.norm_sq < refresh_candidate_sq_)) {
            refresh_candidate_sq_ = candidate.norm_sq;
            refresh_candidate_z_ = candidate.z;
            refresh_candidate_found_at_work_ = current_work;
        }

        if (candidate.norm_sq < phase_best_sq_) {
            phase_best_sq_ = candidate.norm_sq;
            phase_best_z_ = candidate.z;
            phase_best_vector_ = basis_.exact_vector(candidate.z);
            phase_best_found_at_work_ = current_work;
            phase_best_path_ = build_path_records_locked(candidate.z, job);
            radius_decreased = true;
        }

        if (overall_best_sq_ <= 0 || candidate.norm_sq < overall_best_sq_) {
            overall_best_sq_ = candidate.norm_sq;
            overall_best_vector_ = basis_.exact_vector(candidate.z);
        }
    }

    const double value = rollout_value(job.radius_squared, exact_candidates);
    backup_locked(job, value, have_exact, have_exact ? best_exact_score : 0.0);
    release_path_locked(job);

    if (radius_decreased) {
        ++radius_epoch_;
        ++radius_drops_;
        const double ratio = static_cast<double>(std::exp(
            0.5L * Basis::log_positive_mpz(phase_best_sq_) - gso_.log_gh()));
        std::cout << "[BEST] best/GH=" << ratio
                  << " work_nodes=" << current_work
                  << " enum_nodes=" << enumeration_nodes_.load(std::memory_order_relaxed)
                  << "\n" << std::flush;
    }

    propagate_closed_locked(job);
}

void SearchEngine::run_flash() {
    if (nn_enabled_) throw std::runtime_error("run_flash called on NN engine");

    const std::size_t worker_count = std::max<std::size_t>(1, config_.search_threads);
    std::atomic<bool> stop{false};
    std::exception_ptr worker_error;
    std::mutex error_mutex;
    std::condition_variable cv;
    std::mutex cv_mutex;

    auto worker = [&]() {
        try {
            FplllRolloutWorker rollout_worker(basis_);
            for (;;) {
                if (stop.load(std::memory_order_relaxed)) break;

                RolloutJob job;
                bool reserved = false;
                bool should_stop = false;
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    if (budget_reached_locked() || tree_.node(tree_.root_id()).closed) {
                        should_stop = active_rollouts_.load(std::memory_order_relaxed) == 0;
                    } else {
                        reserved = reserve_rollout_locked(job);
                    }
                }

                if (should_stop) {
                    stop.store(true, std::memory_order_relaxed);
                    cv.notify_all();
                    break;
                }

                if (!reserved) {
                    std::unique_lock<std::mutex> wait_lock(cv_mutex);
                    cv.wait_for(wait_lock, std::chrono::milliseconds(1));
                    continue;
                }

                try {
                    RolloutResult rollout = rollout_worker.enumerate_subtree(
                        job.prefix_high_to_low,
                        job.radius_squared,
                        config_.rollout_solutions);

                    std::vector<ExactCandidate> exact;
                    exact.reserve(rollout.stored_coefficients.size());
                    for (const auto& z : rollout.stored_coefficients) {
                        ExactCandidate candidate;
                        candidate.z = z;
                        candidate.norm_sq = basis_.exact_squared_norm(z);
                        exact.push_back(std::move(candidate));
                    }

                    enumeration_nodes_.fetch_add(rollout.enumeration_nodes, std::memory_order_relaxed);
                    exact_candidates_.fetch_add(
                        static_cast<std::uint64_t>(exact.size()), std::memory_order_relaxed);

                    {
                        std::lock_guard<std::mutex> lock(mutex_);
                        for (std::size_t k = 0;
                             k < rollout.nodes_per_level.size() &&
                             k < enumeration_nodes_per_level_.size();
                             ++k) {
                            enumeration_nodes_per_level_[k] += rollout.nodes_per_level[k];
                        }
                        commit_rollout_locked(job, rollout, exact);
                        if ((budget_reached_locked() || tree_.node(tree_.root_id()).closed) &&
                            active_rollouts_.load(std::memory_order_relaxed) == 0) {
                            stop.store(true, std::memory_order_relaxed);
                        }
                    }
                    cv.notify_all();
                } catch (...) {
                    {
                        std::lock_guard<std::mutex> lock(mutex_);
                        if (tree_.node(job.leaf_id).pending_rollout) {
                            release_path_locked(job);
                        }
                    }
                    cv.notify_all();
                    throw;
                }
            }
        } catch (...) {
            {
                std::lock_guard<std::mutex> lock(error_mutex);
                if (!worker_error) worker_error = std::current_exception();
            }
            stop.store(true, std::memory_order_relaxed);
            cv.notify_all();
        }
    };

    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (std::size_t i = 0; i < worker_count; ++i) workers.emplace_back(worker);
    for (auto& thread : workers) thread.join();
    if (worker_error) std::rethrow_exception(worker_error);
}

StatusSnapshot SearchEngine::status_snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    StatusSnapshot status;
    status.phase = phase_index_;
    status.tree_nodes = tree_.size();
    status.enumeration_nodes = enumeration_nodes_.load(std::memory_order_relaxed);
    status.work_nodes = work_nodes_locked();
    status.node_budget = config_.node_budget;
    status.unlimited_nodes = config_.unlimited_nodes;
    status.rollout_jobs = rollout_jobs_.load(std::memory_order_relaxed);
    status.exact_candidates = exact_candidates_.load(std::memory_order_relaxed);
    status.best_found_at_work_node_count = phase_best_found_at_work_;
    status.refresh_candidate_found_at_work_node_count = refresh_candidate_found_at_work_;
    status.root_visits = tree_.node(tree_.root_id()).visits;
    status.radius_drops = radius_drops_;
    status.initial_quality_ratio = input_quality_ratio_;
    status.search_radius_quality_ratio = static_cast<double>(std::exp(
        0.5L * Basis::log_positive_mpz(phase_best_sq_) - gso_.log_gh()));
    status.best_quality_ratio = static_cast<double>(std::exp(
        0.5L * Basis::log_positive_mpz(overall_best_sq_) - gso_.log_gh()));
    status.refresh_candidate_available = !refresh_candidate_z_.empty();
    if (status.refresh_candidate_available) {
        status.refresh_candidate_quality_ratio = static_cast<double>(std::exp(
            0.5L * Basis::log_positive_mpz(refresh_candidate_sq_) - gso_.log_gh()));
    }
    status.budget_reached = budget_reached_locked();
    status.root_closed = tree_.node(tree_.root_id()).closed;
    status.active_rollouts = active_rollouts_.load(std::memory_order_relaxed);
    return status;
}

std::string SearchEngine::diagnostic_status() const {
    const auto s = status_snapshot();
    std::ostringstream out;
    out << "phase=" << s.phase
        << " work_nodes=" << s.work_nodes
        << " tree_nodes=" << s.tree_nodes
        << " enum_nodes=" << s.enumeration_nodes
        << " rollout_jobs=" << s.rollout_jobs
        << " active_rollouts=" << s.active_rollouts
        << " root_visits=" << s.root_visits
        << " root_closed=" << (s.root_closed ? 1 : 0)
        << " budget_reached=" << (s.budget_reached ? 1 : 0);
    return out.str();
}

std::vector<std::int64_t> SearchEngine::refresh_candidate_coefficients() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return refresh_candidate_z_;
}

std::vector<mpz_class> SearchEngine::best_vector() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return overall_best_vector_;
}

std::string SearchEngine::current_basis_text() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return basis_.to_text();
}

std::string SearchEngine::current_basis_packet() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return basis_.to_packet();
}

PhaseSnapshot SearchEngine::snapshot_phase_locked(
    bool refresh_after,
    const BasisRefreshResult* refresh_result) const {
    PhaseSnapshot snapshot;
    snapshot.phase_index = phase_index_;
    snapshot.initial_basis_text = phase_initial_basis_;
    snapshot.tree_nodes = tree_.size();
    snapshot.enumeration_nodes = enumeration_nodes_.load(std::memory_order_relaxed);
    snapshot.work_nodes = work_nodes_locked();
    snapshot.rollout_jobs = rollout_jobs_.load(std::memory_order_relaxed);
    snapshot.exact_candidates = exact_candidates_.load(std::memory_order_relaxed);
    snapshot.best_found_at_work_node_count = phase_best_found_at_work_;
    snapshot.nodes_per_depth = tree_.nodes_per_depth();
    snapshot.enumeration_nodes_per_level = enumeration_nodes_per_level_;
    snapshot.initial_radius_squared = phase_initial_radius_sq_;
    snapshot.initial_radius_over_gh = input_quality_ratio_;
    snapshot.log_gh = gso_.log_gh();
    snapshot.best_squared_norm = phase_best_sq_;
    snapshot.best_coefficients = phase_best_z_;
    snapshot.best_vector = phase_best_vector_;
    snapshot.best_path = best_path_records_locked();
    snapshot.budget_reached = budget_reached_locked();
    snapshot.tree_exhausted = tree_.node(tree_.root_id()).closed;
    snapshot.basis_refresh_after = refresh_after;
    if (refresh_result) {
        snapshot.basis_refresh_changed = refresh_result->changed;
        snapshot.basis_refresh_accepted = refresh_result->accepted;
        snapshot.refresh_potential_before = refresh_result->potential_before;
        snapshot.refresh_potential_after = refresh_result->potential_after;
        snapshot.refreshed_basis_text = refresh_result->basis.to_text();
    }
    return snapshot;
}

bool SearchEngine::refresh_basis_with_best() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_rollouts_.load(std::memory_order_relaxed) != 0) {
        throw std::runtime_error("basis refresh requires zero active rollouts");
    }
    if (refresh_candidate_z_.empty()) {
        throw std::runtime_error("basis refresh requested without a non-basis candidate");
    }

    const auto result = lll_refresh(
        basis_,
        refresh_candidate_z_,
        config_.lll_delta,
        config_.refresh_potential_rel_tolerance);
    if (!result.completed) {
        throw std::runtime_error("LLL refresh failed: " + result.error);
    }

    recorder_.add_phase(snapshot_phase_locked(true, &result));
    phase_snapshot_committed_ = true;
    if (!result.accepted || !result.changed) return false;

    basis_ = result.basis;
    ++phase_index_;
    initialize_episode();
    return true;
}

void SearchEngine::write_results(
    const std::string& result_root,
    const std::string& version,
    const std::string& run_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!phase_snapshot_committed_) {
        recorder_.add_phase(snapshot_phase_locked(false, nullptr));
        phase_snapshot_committed_ = true;
    }
    recorder_.write(result_root, version, run_id);
}

}  // namespace mcts_enum
