#include "mcts/search_engine.hpp"

#include "mcts/basis_refresh.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <exception>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <tuple>

namespace mcts_enum {
namespace {

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

std::vector<float> softmax(const std::vector<float>& logits, double temperature) {
    if (logits.empty()) return {};
    const double tau = std::max(1.0e-6, temperature);
    const float max_logit = *std::max_element(logits.begin(), logits.end());
    std::vector<float> out(logits.size(), 0.0f);
    long double sum = 0.0L;
    for (std::size_t i = 0; i < logits.size(); ++i) {
        const long double e = std::exp(
            (static_cast<long double>(logits[i]) - max_logit) / tau);
        out[i] = static_cast<float>(e);
        sum += e;
    }
    if (!(sum > 0.0L)) {
        const float uniform = 1.0f / static_cast<float>(out.size());
        std::fill(out.begin(), out.end(), uniform);
        return out;
    }
    for (auto& value : out) value = static_cast<float>(value / sum);
    return out;
}

}  // namespace

SearchEngine::SearchEngine(Basis basis, SearchConfig config, bool nn_enabled)
    : basis_(std::move(basis)),
      config_(config),
      nn_enabled_(nn_enabled),
      geometry_(gso_, config_) {
    if (!config_.unlimited_nodes && config_.node_budget == 0) {
        throw std::runtime_error("node_budget must be positive unless unlimited_nodes is enabled");
    }
    if (!(config_.w_m >= 0.0 && config_.w_m <= 1.0)) {
        throw std::runtime_error("w_m must be in [0,1]");
    }
    if (!(config_.dpw > 0.0 && config_.dpw < 1.0)) {
        throw std::runtime_error("dpw must be in (0,1)");
    }
    if (!(config_.policy_mix >= 0.0 && config_.policy_mix <= 1.0)) {
        throw std::runtime_error("policy_mix must be in [0,1]");
    }
    if (config_.radius_global_update_interval == 0) {
        throw std::runtime_error("radius_global_update_interval must be positive");
    }
    recorder_.set_nn_enabled(nn_enabled_);
    recorder_.set_search_config(config_);
    initialize_episode();
}

SearchEngine SearchEngine::from_packet(
    const std::string& packet,
    const SearchConfig& config,
    bool nn_enabled) {
    return SearchEngine(Basis::from_packet(packet), config, nn_enabled);
}

void SearchEngine::initialize_episode() {
    gso_.recompute(basis_);
    gso_features_cache_ = gso_.normalized_features();
    r0_sq_ = basis_.first_vector_squared_norm();
    if (r0_sq_ <= 0) throw std::runtime_error("first basis vector is zero");
    best_sq_ = r0_sq_;
    best_z_ = basis_.first_vector_coefficients();
    refresh_candidate_sq_ = 0;
    refresh_candidate_z_.clear();
    refresh_candidate_found_at_node_count_ = 0;
    best_terminal_node_id_ = -1;
    best_found_at_node_count_ = 0;
    best_path_snapshot_.clear();
    best_score_ = 0.0;

    const long double log_b1 = 0.5L * Basis::log_positive_mpz(r0_sq_);
    input_quality_ratio_ = static_cast<double>(std::exp(log_b1 - gso_.log_gh()));
    if (input_quality_ratio_ > config_.quality_gate) {
        throw std::runtime_error(
            "input basis rejected by quality gate: b1/GH exceeds configured threshold");
    }

    tree_.reset(basis_.dimension());
    phase_initial_basis_ = basis_.to_text();
    pending_.clear();
    progress_epoch_ = 0;
    radius_epoch_ = 0;
    refresh_node_baseline_ = static_cast<std::uint64_t>(tree_.size());
    radius_drops_since_global_update_ = 0;
    initialize_node_geometry(tree_.node(tree_.root_id()));
    if (!nn_enabled_ && !tree_.node(tree_.root_id()).pruned) {
        initialize_flash_policy(tree_.node(tree_.root_id()));
    }
    ++progress_epoch_;
}

bool SearchEngine::node_budget_reached() const noexcept {
    return !config_.unlimited_nodes && tree_.size() >= config_.node_budget;
}

bool SearchEngine::can_create_node() const noexcept {
    return config_.unlimited_nodes || tree_.size() < config_.node_budget;
}

bool SearchEngine::finished() const noexcept {
    return (node_budget_reached() || tree_.node(tree_.root_id()).closed) &&
           pending_.empty();
}

std::size_t SearchEngine::pending_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return pending_.size();
}

std::string SearchEngine::diagnostic_status() const {
    std::lock_guard<std::mutex> lock(mutex_);
    const Node& root = tree_.node(tree_.root_id());
    std::size_t active_edges = 0;
    std::size_t expanded_edges = 0;
    std::size_t open_children = 0;
    std::size_t pending_children = 0;
    std::size_t invalid_child_links = 0;
    for (const auto& edge : root.edges) {
        if (!edge.active) continue;
        ++active_edges;
        if (!edge.expanded || edge.child_id < 0) continue;
        ++expanded_edges;
        if (static_cast<std::size_t>(edge.child_id) >= tree_.size()) {
            ++invalid_child_links;
            continue;
        }
        const Node& child = tree_.node(edge.child_id);
        if (child.pending) ++pending_children;
        if (!child.closed && !child.pruned) ++open_children;
    }

    std::ostringstream out;
    out << "nodes=" << tree_.size()
        << " progress_epoch=" << progress_epoch_
        << " pending=" << pending_.size()
        << " budget_reached=" << (node_budget_reached() ? 1 : 0)
        << " root_closed=" << (root.closed ? 1 : 0)
        << " root_pruned=" << (root.pruned ? 1 : 0)
        << " root_initialized=" << (root.initialized ? 1 : 0)
        << " root_pending=" << (root.pending ? 1 : 0)
        << " root_visits=" << root.visits
        << " root_edges=" << root.edges.size()
        << " root_active_edges=" << active_edges
        << " root_expanded_edges=" << expanded_edges
        << " root_open_children=" << open_children
        << " root_pending_children=" << pending_children
        << " root_invalid_child_links=" << invalid_child_links
        << " radius_drops_since_global_update="
        << radius_drops_since_global_update_;
    return out.str();
}

StatusSnapshot SearchEngine::status_snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    StatusSnapshot status;
    const Node& root = tree_.node(tree_.root_id());
    status.phase = phase_index_;
    status.nodes = static_cast<std::uint64_t>(tree_.size());
    status.node_budget = config_.node_budget;
    status.unlimited_nodes = config_.unlimited_nodes;
    status.best_found_at_node_count = best_found_at_node_count_;
    status.refresh_candidate_found_at_node_count = refresh_candidate_found_at_node_count_;
    status.nodes_since_refresh = status.nodes - refresh_node_baseline_;
    status.root_visits = root.visits;
    status.progress_epoch = progress_epoch_;
    status.pending = pending_.size();
    status.radius_drops_since_global_update = radius_drops_since_global_update_;
    status.initial_quality_ratio = input_quality_ratio_;
    status.best_score = best_score_;
    status.budget_reached = node_budget_reached();
    status.root_closed = root.closed;
    if (best_sq_ > 0) {
        const long double log_best = 0.5L * Basis::log_positive_mpz(best_sq_);
        status.best_quality_ratio = static_cast<double>(std::exp(log_best - gso_.log_gh()));
    }
    status.refresh_candidate_available = !refresh_candidate_z_.empty();
    if (status.refresh_candidate_available && refresh_candidate_sq_ > 0) {
        const long double log_candidate =
            0.5L * Basis::log_positive_mpz(refresh_candidate_sq_);
        status.refresh_candidate_quality_ratio =
            static_cast<double>(std::exp(log_candidate - gso_.log_gh()));
    }
    return status;
}

long double SearchEngine::radius_sq_scaled() const {
    return Basis::scaled_positive_mpz(best_sq_, 2 * gso_.scale_exp2());
}

std::vector<std::int64_t> SearchEngine::reconstruct_coefficients(
    std::int64_t node_id) const {
    std::vector<std::int64_t> z(basis_.dimension(), 0);
    std::int64_t current = node_id;
    while (current > 0) {
        const Node& child = tree_.node(current);
        const Node& parent = tree_.node(child.parent_id);
        if (parent.next_k < 0 || parent.next_k >= basis_.dimension()) {
            throw std::runtime_error("invalid tree coefficient index");
        }
        z[static_cast<std::size_t>(parent.next_k)] = child.action_from_parent;
        current = child.parent_id;
    }
    return z;
}

void SearchEngine::apply_node_geometry(
    Node& node,
    GeometryInfo geometry,
    std::vector<CandidateInfo> candidates) {
    node.geometry = std::move(geometry);
    node.geometry_ready = true;
    node.expanding = false;
    node.pruned = node.geometry.pruned;
    node.edges.clear();
    node.candidate_features.clear();
    if (node.pruned) {
        node.initialized = true;
        node.closed = true;
        return;
    }

    node.edges.reserve(candidates.size());
    node.candidate_features.reserve(candidates.size() * 3);
    for (const auto& candidate : candidates) {
        EdgeStats edge;
        edge.action = candidate.action;
        edge.active = true;
        node.edges.push_back(edge);
        node.candidate_features.push_back(candidate.normalized_offset);
        node.candidate_features.push_back(candidate.normalized_abs_offset);
        node.candidate_features.push_back(candidate.se_rank);
    }
    if (node.edges.empty()) {
        node.pruned = true;
        node.initialized = true;
        node.closed = true;
    }
}

void SearchEngine::initialize_node_geometry(Node& node) {
    if (node.terminal) {
        node.geometry_ready = true;
        node.initialized = true;
        return;
    }
    const auto z = reconstruct_coefficients(node.id);
    auto geometry = geometry_.legal_interval(
        node.next_k, node.rho, radius_sq_scaled(), z);
    auto candidates = geometry_.candidates(geometry);
    apply_node_geometry(node, std::move(geometry), std::move(candidates));
}

std::vector<float> SearchEngine::global_features(const Node& node) const {
    const long double radius_sq = radius_sq_scaled();
    const long double radius = std::sqrt(std::max(radius_sq, 0.0L));
    const long double gh = std::max(gso_.gh_scaled(), 1.0e-30L);
    const long double depth = static_cast<long double>(basis_.dimension() - 1 - node.next_k);
    const long double n = static_cast<long double>(basis_.dimension());
    const long double frac_center = node.terminal
        ? 0.0L
        : node.geometry.center - std::nearbyint(node.geometry.center);
    const long double active_count = std::count_if(
        node.edges.begin(), node.edges.end(), [](const EdgeStats& e) { return e.active; });
    return {
        static_cast<float>(node.next_k < 0 ? 1.0L : static_cast<long double>(node.next_k + 1) / n),
        static_cast<float>(radius_sq > 0.0L ? node.rho / radius_sq : 0.0L),
        static_cast<float>(radius / gh),
        static_cast<float>(depth / n),
        static_cast<float>(frac_center),
        static_cast<float>(std::log1p(node.terminal ? 0.0L : node.geometry.delta)),
        static_cast<float>(std::log1p(active_count)),
        static_cast<float>(node.visits > 0 ? std::log1p(static_cast<double>(node.visits)) : 0.0)
    };
}

std::vector<float> SearchEngine::recent_residual_features(std::int64_t node_id) const {
    std::vector<float> values;
    values.reserve(config_.recent_residual_count);
    std::int64_t current = node_id;
    while (current > 0 && values.size() < config_.recent_residual_count) {
        const Node& child = tree_.node(current);
        const Node& parent = tree_.node(child.parent_id);
        const long double delta = std::max(parent.geometry.delta, 1.0e-30L);
        const long double y = static_cast<long double>(child.action_from_parent) - parent.geometry.center;
        values.push_back(static_cast<float>(y / delta));
        current = child.parent_id;
    }
    std::reverse(values.begin(), values.end());
    if (values.size() < config_.recent_residual_count) {
        values.insert(values.begin(), config_.recent_residual_count - values.size(), 0.0f);
    }
    return values;
}

EvalRequest SearchEngine::make_request(const Node& node, std::uint64_t request_id) const {
    EvalRequest request;
    request.request_id = request_id;
    request.node_id = node.id;
    request.global_features = global_features(node);
    request.recent_residuals = recent_residual_features(node.id);
    request.candidate_features = node.candidate_features;
    request.candidate_count = node.edges.size();
    return request;
}

std::size_t SearchEngine::progressive_widening_limit(const Node& node) const {
    const std::size_t active = static_cast<std::size_t>(std::count_if(
        node.edges.begin(), node.edges.end(), [](const EdgeStats& edge) { return edge.active; }));
    if (active == 0) return 0;
    const double visits = static_cast<double>(std::max<std::uint64_t>(1, node.visits));
    const std::size_t allowed = static_cast<std::size_t>(std::floor(
        config_.cpw * std::pow(visits, config_.dpw)));
    return std::max<std::size_t>(1, std::min(active, allowed));
}

std::size_t SearchEngine::next_unexpanded_edge(const Node& node) const {
    for (std::size_t i = 0; i < node.edges.size(); ++i) {
        if (node.edges[i].active && !node.edges[i].expanded) return i;
    }
    return node.edges.size();
}

std::size_t SearchEngine::choose_edge(const Node& node) const {
    const double sqrt_parent = std::sqrt(static_cast<double>(std::max<std::uint64_t>(1, node.visits)));
    double best_score = -std::numeric_limits<double>::infinity();
    std::size_t best = node.edges.size();
    for (std::size_t i = 0; i < node.edges.size(); ++i) {
        const EdgeStats& edge = node.edges[i];
        if (!edge.active || !edge.expanded || edge.child_id < 0) continue;
        const Node& child = tree_.node(edge.child_id);
        if (child.pending || child.expanding || child.closed) continue;
        const double m_effective = edge.has_exact_m ? edge.m : edge.q;
        const double exploitation = (1.0 - config_.w_m) * edge.q + config_.w_m * m_effective;
        const double exploration = config_.lambda_puct * edge.prior * sqrt_parent /
                                   (1.0 + static_cast<double>(edge.visits));
        const double score = exploitation + exploration;
        if (score > best_score ||
            (score == best_score &&
             (best >= node.edges.size() || edge.prior > node.edges[best].prior))) {
            best_score = score;
            best = i;
        }
    }
    return best;
}

std::vector<std::pair<std::int64_t, std::size_t>>
SearchEngine::select_path_to_leaf(
    std::int64_t* leaf_id, bool* created_node, bool defer_geometry) {
    std::vector<std::pair<std::int64_t, std::size_t>> path;
    std::int64_t current = tree_.root_id();
    *created_node = false;

    for (;;) {
        Node& node = tree_.node(current);
        if (node.closed) {
            *leaf_id = -1;
            return path;
        }
        if (node.terminal || node.pruned || !node.initialized || node.pending || node.expanding) {
            *leaf_id = current;
            return path;
        }

        const std::size_t limit = progressive_widening_limit(node);
        const std::size_t expanded = static_cast<std::size_t>(std::count_if(
            node.edges.begin(), node.edges.end(), [](const EdgeStats& edge) {
                return edge.active && edge.expanded;
            }));

        std::size_t edge_index = node.edges.size();
        if (can_create_node() && expanded < limit) {
            edge_index = next_unexpanded_edge(node);
            if (edge_index < node.edges.size()) {
                const EdgeStats& edge = node.edges[edge_index];
                const long double child_rho = geometry_.next_rho(
                    node.next_k, node.rho, node.geometry.center, edge.action);
                const std::int64_t child_id = tree_.create_child(current, edge_index, child_rho);
                ++progress_epoch_;
                path.emplace_back(current, edge_index);
                Node& child = tree_.node(child_id);
                if (!child.terminal && !defer_geometry) initialize_node_geometry(child);
                *leaf_id = child_id;
                *created_node = true;
                return path;
            }
        }

        edge_index = choose_edge(node);
        if (edge_index >= node.edges.size()) {
            const bool has_pending_child = std::any_of(
                node.edges.begin(), node.edges.end(), [&](const EdgeStats& edge) {
                    return edge.active && edge.expanded && edge.child_id >= 0 &&
                           (tree_.node(edge.child_id).pending ||
                            tree_.node(edge.child_id).expanding);
                });
            if (has_pending_child) {
                *leaf_id = -1;
                return path;
            }

            if (!can_create_node()) {
                *leaf_id = -1;
                return path;
            }

            edge_index = next_unexpanded_edge(node);
            if (edge_index < node.edges.size()) {
                const EdgeStats& edge = node.edges[edge_index];
                const long double child_rho = geometry_.next_rho(
                    node.next_k, node.rho, node.geometry.center, edge.action);
                const std::int64_t child_id = tree_.create_child(current, edge_index, child_rho);
                ++progress_epoch_;
                path.emplace_back(current, edge_index);
                Node& child = tree_.node(child_id);
                if (!child.terminal && !defer_geometry) initialize_node_geometry(child);
                *leaf_id = child_id;
                *created_node = true;
                return path;
            }

            node.closed = true;
            ++progress_epoch_;
            *leaf_id = -1;
            return path;
        }
        path.emplace_back(current, edge_index);
        current = node.edges[edge_index].child_id;
    }
}

void SearchEngine::set_policy_and_value(
    Node& node, const std::vector<float>& logits, float value, bool count_progress) {
    if (logits.size() != node.edges.size()) {
        throw std::runtime_error("policy logit count does not match legal action count");
    }
    if (node.edges.empty()) {
        node.initialized = true;
        node.pending = false;
        node.value_estimate = value;
        if (count_progress) ++progress_epoch_;
        return;
    }

    auto probabilities = softmax(logits, 1.0);
    long double active_sum = 0.0L;
    for (std::size_t i = 0; i < probabilities.size(); ++i) {
        if (!node.edges[i].active) probabilities[i] = 0.0f;
        active_sum += probabilities[i];
    }
    if (!(active_sum > 0.0L)) {
        const std::size_t active_count = static_cast<std::size_t>(std::count_if(
            node.edges.begin(), node.edges.end(), [](const EdgeStats& e) { return e.active; }));
        const float uniform = active_count ? 1.0f / static_cast<float>(active_count) : 0.0f;
        for (std::size_t i = 0; i < probabilities.size(); ++i) {
            probabilities[i] = node.edges[i].active ? uniform : 0.0f;
        }
    } else {
        for (auto& p : probabilities) p = static_cast<float>(p / active_sum);
    }

    std::vector<std::size_t> policy_order(node.edges.size());
    std::iota(policy_order.begin(), policy_order.end(), 0);
    std::stable_sort(policy_order.begin(), policy_order.end(), [&](std::size_t a, std::size_t b) {
        return logits[a] > logits[b];
    });
    std::vector<double> policy_rank(node.edges.size(), 1.0);
    const double denom = node.edges.size() > 1 ? static_cast<double>(node.edges.size() - 1) : 1.0;
    for (std::size_t rank = 0; rank < policy_order.size(); ++rank) {
        policy_rank[policy_order[rank]] = static_cast<double>(rank) / denom;
    }

    struct Bundle {
        EdgeStats edge;
        float f0 = 0.0f;
        float f1 = 0.0f;
        float f2 = 0.0f;
        double proposal = 0.0;
    };
    std::vector<Bundle> bundles;
    bundles.reserve(node.edges.size());
    for (std::size_t i = 0; i < node.edges.size(); ++i) {
        node.edges[i].prior = probabilities[i];
        node.edges[i].policy_prior = probabilities[i];
        const float se_rank = node.candidate_features[i * 3 + 2];
        Bundle bundle;
        bundle.edge = node.edges[i];
        bundle.f0 = node.candidate_features[i * 3];
        bundle.f1 = node.candidate_features[i * 3 + 1];
        bundle.f2 = se_rank;
        bundle.proposal = config_.policy_mix * policy_rank[i] +
                          (1.0 - config_.policy_mix) * static_cast<double>(se_rank);
        if (!bundle.edge.active) bundle.proposal += 10.0;
        bundles.push_back(std::move(bundle));
    }
    std::stable_sort(bundles.begin(), bundles.end(), [](const Bundle& a, const Bundle& b) {
        if (a.proposal != b.proposal) return a.proposal < b.proposal;
        return a.edge.action < b.edge.action;
    });

    node.edges.clear();
    node.candidate_features.clear();
    for (auto& bundle : bundles) {
        node.edges.push_back(std::move(bundle.edge));
        node.candidate_features.push_back(bundle.f0);
        node.candidate_features.push_back(bundle.f1);
        node.candidate_features.push_back(bundle.f2);
    }
    node.value_estimate = value;
    node.initialized = true;
    node.pending = false;
    if (count_progress) ++progress_epoch_;
}

void SearchEngine::initialize_flash_policy(Node& node) {
    std::vector<float> logits(node.edges.size(), 0.0f);
    for (std::size_t i = 0; i < node.edges.size(); ++i) {
        logits[i] = -2.0f * node.candidate_features[i * 3 + 1];
    }
    set_policy_and_value(node, logits, 0.0f);
}

void SearchEngine::backup(
    const std::vector<std::pair<std::int64_t, std::size_t>>& path,
    double value,
    bool exact_terminal) {
    tree_.node(tree_.root_id()).visits += 1;
    if (exact_terminal) {
        Node& root = tree_.node(tree_.root_id());
        root.has_terminal_score = true;
        root.best_terminal_score = std::max(root.best_terminal_score, value);
    }

    std::int64_t last_child = tree_.root_id();
    for (const auto& [parent_id, edge_index] : path) {
        Node& parent = tree_.node(parent_id);
        EdgeStats& edge = parent.edges.at(edge_index);
        edge.visits += 1;
        edge.w += value;
        edge.q = edge.w / static_cast<double>(edge.visits);
        if (exact_terminal) {
            edge.has_exact_m = true;
            edge.m = std::max(edge.m, value);
            parent.has_terminal_score = true;
            parent.best_terminal_score = std::max(parent.best_terminal_score, value);
        }
        last_child = edge.child_id;
        if (last_child >= 0) tree_.node(last_child).visits += 1;
    }
    if (exact_terminal && last_child >= 0) {
        Node& leaf = tree_.node(last_child);
        leaf.has_terminal_score = true;
        leaf.best_terminal_score = std::max(leaf.best_terminal_score, value);
    }
    ++progress_epoch_;
}

void SearchEngine::tighten_tree_after_radius_update() {
    const long double radius = radius_sq_scaled();
    const std::size_t node_total = tree_.size();

    auto process_range = [&](std::size_t begin, std::size_t end) {
        for (std::size_t id = begin; id < end; ++id) {
            Node& node = tree_.node(static_cast<std::int64_t>(id));
            if (node.terminal || !node.geometry_ready || node.edges.empty()) continue;
            const auto z = reconstruct_coefficients(node.id);
            node.geometry = geometry_.legal_interval(node.next_k, node.rho, radius, z);
            if (node.geometry.pruned) {
                node.pruned = true;
                node.closed = true;
                for (auto& edge : node.edges) {
                    edge.active = false;
                    edge.prior = 0.0;
                    edge.policy_prior = 0.0;
                }
                continue;
            }
            node.pruned = false;
            long double prior_sum = 0.0L;
            long double policy_prior_sum = 0.0L;
            std::size_t active_count = 0;
            for (std::size_t i = 0; i < node.edges.size(); ++i) {
                auto& edge = node.edges[i];
                edge.active = edge.action >= node.geometry.lo && edge.action <= node.geometry.hi;
                if (edge.active) {
                    prior_sum += edge.prior;
                    policy_prior_sum += edge.policy_prior;
                    ++active_count;
                }
                const long double offset = static_cast<long double>(edge.action) - node.geometry.center;
                const long double delta = std::max(node.geometry.delta, 1.0e-30L);
                node.candidate_features[i * 3] = static_cast<float>(offset / delta);
                node.candidate_features[i * 3 + 1] = std::fabs(node.candidate_features[i * 3]);
            }

            const double uniform = active_count > 0
                ? 1.0 / static_cast<double>(active_count)
                : 0.0;
            for (auto& edge : node.edges) {
                if (!edge.active) {
                    edge.prior = 0.0;
                    edge.policy_prior = 0.0;
                    continue;
                }
                edge.prior = prior_sum > 0.0L
                    ? static_cast<double>(static_cast<long double>(edge.prior) / prior_sum)
                    : uniform;
                edge.policy_prior = policy_prior_sum > 0.0L
                    ? static_cast<double>(static_cast<long double>(edge.policy_prior) / policy_prior_sum)
                    : uniform;
            }
        }
    };

    const std::size_t workers = std::max<std::size_t>(
        1, std::min<std::size_t>(node_total, static_cast<std::size_t>(config_.search_threads)));
    if (workers == 1 || node_total < 4096) {
        process_range(0, node_total);
    } else {
        std::vector<std::thread> threads;
        threads.reserve(workers);
        const std::size_t chunk = (node_total + workers - 1) / workers;
        for (std::size_t worker = 0; worker < workers; ++worker) {
            const std::size_t begin = worker * chunk;
            const std::size_t end = std::min(node_total, begin + chunk);
            if (begin >= end) break;
            threads.emplace_back(process_range, begin, end);
        }
        for (auto& thread : threads) thread.join();
    }

    for (auto& [request_id, pending] : pending_) {
        (void)request_id;
        bool valid = pending.node_id >= 0 &&
                     static_cast<std::size_t>(pending.node_id) < tree_.size();
        if (valid) {
            const Node& leaf = tree_.node(pending.node_id);
            valid = !leaf.pruned && !leaf.closed;
        }
        if (valid) {
            for (const auto& [parent_id, edge_index] : pending.path) {
                if (parent_id < 0 || static_cast<std::size_t>(parent_id) >= tree_.size()) {
                    valid = false;
                    break;
                }
                const Node& parent = tree_.node(parent_id);
                if (parent.pruned || parent.closed || edge_index >= parent.edges.size() ||
                    !parent.edges[edge_index].active) {
                    valid = false;
                    break;
                }
            }
        }
        pending.valid = valid;
    }
    ++progress_epoch_;
}

void SearchEngine::on_radius_decreased() {
    ++radius_epoch_;
    ++radius_drops_since_global_update_;
    if (radius_drops_since_global_update_ < config_.radius_global_update_interval) {
        return;
    }
    tighten_tree_after_radius_update();
    radius_drops_since_global_update_ = 0;
}

void SearchEngine::commit_terminal(
    std::int64_t node_id,
    const std::vector<std::pair<std::int64_t, std::size_t>>& path,
    const std::vector<std::int64_t>& z,
    const mpz_class& norm_sq) {
    tree_.node(node_id).closed = true;
    if (norm_sq == 0) {
        backup(path, 0.0, false);
        return;
    }
    if (!is_single_basis_vector(z) &&
        (refresh_candidate_sq_ == 0 || norm_sq < refresh_candidate_sq_)) {
        refresh_candidate_sq_ = norm_sq;
        refresh_candidate_z_ = z;
        refresh_candidate_found_at_node_count_ =
            static_cast<std::uint64_t>(tree_.size());
    }
    const double score = terminal_score_from_norm(norm_sq);
    backup(path, score, true);
    if (norm_sq < best_sq_) {
        // Snapshot the exact legal action-space width at each level under the
        // incumbent radius that this terminal vector actually beat.
        const long double discovery_radius_sq = radius_sq_scaled();
        std::vector<std::uint64_t> action_space_sizes;
        action_space_sizes.reserve(path.size());
        for (const auto& [parent_id, edge_index] : path) {
            (void)edge_index;
            const Node& parent = tree_.node(parent_id);
            const auto parent_z = reconstruct_coefficients(parent_id);
            const auto geom = geometry_.legal_interval(
                parent.next_k, parent.rho, discovery_radius_sq, parent_z);
            std::uint64_t count = 0;
            if (!geom.pruned && geom.hi >= geom.lo) {
                const __int128 width =
                    static_cast<__int128>(geom.hi) -
                    static_cast<__int128>(geom.lo) + 1;
                if (width > 0) count = static_cast<std::uint64_t>(width);
            }
            action_space_sizes.push_back(count);
        }

        best_sq_ = norm_sq;
        best_z_ = z;
        best_terminal_node_id_ = node_id;
        best_found_at_node_count_ = static_cast<std::uint64_t>(tree_.size());
        best_score_ = score;
        best_path_snapshot_ = best_path_records();

        const long double log_best = 0.5L * Basis::log_positive_mpz(best_sq_);
        const double ratio = static_cast<double>(std::exp(log_best - gso_.log_gh()));
        const std::uint64_t total_nodes =
            completed_phase_nodes_ + static_cast<std::uint64_t>(tree_.size());
        std::ostringstream message;
        message << std::fixed << std::setprecision(8);
        message << "[BEST] best/GH=" << ratio
                << " total_nodes=" << total_nodes << '\n';
        message << "       z(z_1->z_n)=[";
        for (std::size_t i = 0; i < z.size(); ++i) {
            if (i) message << ',';
            message << z[i];
        }
        message << "]\n";
        message << "       |A|(z_n->z_1)=[";
        for (std::size_t i = 0; i < action_space_sizes.size(); ++i) {
            if (i) message << ',';
            message << action_space_sizes[i];
        }
        message << "]\n";
        std::cout << message.str() << std::flush;

        on_radius_decreased();
    }
}

void SearchEngine::evaluate_terminal(
    std::int64_t node_id,
    const std::vector<std::pair<std::int64_t, std::size_t>>& path) {
    const auto z = reconstruct_coefficients(node_id);
    const mpz_class norm_sq = basis_.exact_squared_norm(z);
    commit_terminal(node_id, path, z, norm_sq);
}

double SearchEngine::terminal_score_from_norm(const mpz_class& norm_sq) const {
    if (norm_sq <= 0) return 0.0;
    const long double value =
        0.5L * static_cast<long double>(basis_.dimension()) *
        (Basis::log_positive_mpz(r0_sq_) - Basis::log_positive_mpz(norm_sq));
    return static_cast<double>(value);
}

void SearchEngine::run_flash() {
    if (nn_enabled_) throw std::runtime_error("run_flash called on NN-enabled engine");

    const std::size_t worker_count = std::max<std::size_t>(
        1, static_cast<std::size_t>(config_.search_threads));
    std::atomic<bool> stop{false};
    std::atomic<std::uint64_t> active_terminal_jobs{0};
    std::atomic<std::uint64_t> no_progress{0};
    std::exception_ptr worker_error;
    std::mutex worker_error_mutex;
    const std::uint64_t no_progress_limit = config_.unlimited_nodes
        ? 1'000'000ULL
        : std::max<std::uint64_t>(4096, config_.node_budget * 32 + 1024);

    auto worker = [&]() {
        try {
        while (!stop.load(std::memory_order_relaxed)) {
            std::int64_t leaf_id = -1;
            std::vector<std::pair<std::int64_t, std::size_t>> path;
            std::vector<std::int64_t> terminal_z;
            bool terminal_job = false;

            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (node_budget_reached() || tree_.node(tree_.root_id()).closed) {
                    stop.store(true, std::memory_order_relaxed);
                    break;
                }

                bool created = false;
                path = select_path_to_leaf(&leaf_id, &created);
                if (leaf_id >= 0) {
                    Node& leaf = tree_.node(leaf_id);
                    if (leaf.terminal) {
                        if (!leaf.pending) {
                            leaf.pending = true;
                            terminal_z = reconstruct_coefficients(leaf_id);
                            terminal_job = true;
                            active_terminal_jobs.fetch_add(1, std::memory_order_relaxed);
                        }
                    } else if (leaf.pruned) {
                        backup(path, 0.0, false);
                        no_progress.store(0, std::memory_order_relaxed);
                    } else if (!leaf.initialized) {
                        initialize_flash_policy(leaf);
                        backup(path, leaf.value_estimate, false);
                        no_progress.store(0, std::memory_order_relaxed);
                    }
                }
            }

            if (terminal_job) {
                const mpz_class norm_sq = basis_.exact_squared_norm(terminal_z);
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    Node& leaf = tree_.node(leaf_id);
                    leaf.pending = false;
                    commit_terminal(leaf_id, path, terminal_z, norm_sq);
                    no_progress.store(0, std::memory_order_relaxed);
                }
                active_terminal_jobs.fetch_sub(1, std::memory_order_relaxed);
                continue;
            }

            if (leaf_id < 0) {
                if (active_terminal_jobs.load(std::memory_order_relaxed) == 0) {
                    const auto count = no_progress.fetch_add(1, std::memory_order_relaxed) + 1;
                    if (count >= no_progress_limit) {
                        throw std::runtime_error(
                            "Flash MCTS stalled without reaching node budget or exhausting the tree");
                    }
                }
                std::this_thread::yield();
            }
        }
        } catch (...) {
            {
                std::lock_guard<std::mutex> error_lock(worker_error_mutex);
                if (!worker_error) worker_error = std::current_exception();
            }
            stop.store(true, std::memory_order_relaxed);
        }
    };

    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (std::size_t i = 0; i < worker_count; ++i) workers.emplace_back(worker);
    for (auto& thread : workers) thread.join();
    if (worker_error) std::rethrow_exception(worker_error);
}

std::vector<EvalRequest> SearchEngine::collect_inference_batch(std::size_t max_batch) {
    if (!nn_enabled_) throw std::runtime_error("inference batch requested in flash mode");
    if (max_batch == 0) return {};

    struct GeometryJob {
        std::int64_t node_id = -1;
        int k = -1;
        long double rho = 0.0L;
        long double radius_sq = 0.0L;
        std::uint64_t radius_epoch = 0;
        std::vector<std::int64_t> z;
    };
    struct TerminalJob {
        std::int64_t node_id = -1;
        std::vector<std::pair<std::int64_t, std::size_t>> path;
        std::vector<std::int64_t> z;
    };

    std::vector<EvalRequest> batch;
    batch.reserve(max_batch);
    const std::size_t worker_count = std::max<std::size_t>(
        1, std::min<std::size_t>(max_batch, static_cast<std::size_t>(config_.search_threads)));
    std::atomic<std::uint64_t> active_native_jobs{0};
    std::atomic<bool> stop{false};
    std::exception_ptr worker_error;
    std::mutex worker_error_mutex;
    std::mutex work_wait_mutex;
    std::condition_variable work_wait_cv;

    auto worker = [&]() {
        try {
            const std::size_t dimension_guard =
                static_cast<std::size_t>(std::max(1, basis_.dimension())) * 64;
            const std::size_t max_attempts = std::max<std::size_t>(4096, dimension_guard);
            std::size_t attempts = 0;

            while (!stop.load(std::memory_order_relaxed) && attempts++ < max_attempts) {
                GeometryJob geometry_job;
                TerminalJob terminal_job;
                bool have_geometry_job = false;
                bool have_terminal_job = false;
                bool made_exact_progress = false;
                bool wait_for_native_work = false;

                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    if (batch.size() >= max_batch || node_budget_reached() ||
                        tree_.node(tree_.root_id()).closed) {
                        if (batch.size() >= max_batch || finished()) {
                            stop.store(true, std::memory_order_relaxed);
                        }
                        break;
                    }

                    std::int64_t leaf_id = -1;
                    bool created = false;
                    auto path = select_path_to_leaf(&leaf_id, &created, true);
                    if (leaf_id < 0) {
                        if (active_native_jobs.load(std::memory_order_relaxed) == 0) {
                            break;
                        }
                        wait_for_native_work = true;
                    } else {
                        Node& leaf = tree_.node(leaf_id);
                        if (leaf.terminal) {
                            if (!leaf.expanding && !leaf.pending && !leaf.closed) {
                                leaf.expanding = true;
                                terminal_job.node_id = leaf_id;
                                terminal_job.path = std::move(path);
                                terminal_job.z = reconstruct_coefficients(leaf_id);
                                have_terminal_job = true;
                                active_native_jobs.fetch_add(1, std::memory_order_relaxed);
                            }
                        } else if (leaf.pruned) {
                            backup(path, 0.0, false);
                            made_exact_progress = true;
                        } else if (!leaf.geometry_ready) {
                            if (!leaf.expanding && !leaf.closed) {
                                leaf.expanding = true;
                                geometry_job.node_id = leaf_id;
                                geometry_job.k = leaf.next_k;
                                geometry_job.rho = leaf.rho;
                                geometry_job.radius_sq = radius_sq_scaled();
                                geometry_job.radius_epoch = radius_epoch_;
                                geometry_job.z = reconstruct_coefficients(leaf_id);
                                have_geometry_job = true;
                                active_native_jobs.fetch_add(1, std::memory_order_relaxed);
                            }
                        } else if (!leaf.initialized && !leaf.pending && !leaf.expanding) {
                            const std::uint64_t request_id = next_request_id_++;
                            leaf.pending = true;
                            PendingEvaluation pending;
                            pending.request_id = request_id;
                            pending.node_id = leaf_id;
                            pending.path = std::move(path);
                            pending_.emplace(request_id, std::move(pending));
                            batch.push_back(make_request(leaf, request_id));
                            ++progress_epoch_;
                            made_exact_progress = true;
                            if (batch.size() >= max_batch) {
                                stop.store(true, std::memory_order_relaxed);
                            }
                        }
                    }
                }

                if (have_geometry_job) {
                    auto geometry = geometry_.legal_interval(
                        geometry_job.k,
                        geometry_job.rho,
                        geometry_job.radius_sq,
                        geometry_job.z);
                    auto candidates = geometry_.candidates(geometry);
                    {
                        std::lock_guard<std::mutex> lock(mutex_);
                        Node& node = tree_.node(geometry_job.node_id);
                        if (node.closed || node.pruned ||
                            geometry_job.radius_epoch != radius_epoch_) {
                            node.expanding = false;
                        } else {
                            apply_node_geometry(node, std::move(geometry), std::move(candidates));
                            ++progress_epoch_;
                        }
                    }
                    active_native_jobs.fetch_sub(1, std::memory_order_relaxed);
                    work_wait_cv.notify_all();
                    continue;
                }

                if (have_terminal_job) {
                    const mpz_class norm_sq = basis_.exact_squared_norm(terminal_job.z);
                    {
                        std::lock_guard<std::mutex> lock(mutex_);
                        Node& leaf = tree_.node(terminal_job.node_id);
                        leaf.expanding = false;
                        if (!leaf.closed) {
                            commit_terminal(
                                terminal_job.node_id,
                                terminal_job.path,
                                terminal_job.z,
                                norm_sq);
                        }
                    }
                    active_native_jobs.fetch_sub(1, std::memory_order_relaxed);
                    work_wait_cv.notify_all();
                    continue;
                }

                if (wait_for_native_work) {
                    std::unique_lock<std::mutex> wait_lock(work_wait_mutex);
                    work_wait_cv.wait_for(wait_lock, std::chrono::milliseconds(1));
                    continue;
                }

                if (!made_exact_progress) {
                    std::this_thread::yield();
                }
            }
        } catch (...) {
            {
                std::lock_guard<std::mutex> error_lock(worker_error_mutex);
                if (!worker_error) worker_error = std::current_exception();
            }
            stop.store(true, std::memory_order_relaxed);
            work_wait_cv.notify_all();
        }
    };

    if (worker_count == 1) {
        worker();
    } else {
        std::vector<std::thread> workers;
        workers.reserve(worker_count);
        for (std::size_t i = 0; i < worker_count; ++i) workers.emplace_back(worker);
        for (auto& thread : workers) thread.join();
    }
    if (worker_error) std::rethrow_exception(worker_error);

    std::sort(batch.begin(), batch.end(), [](const EvalRequest& a, const EvalRequest& b) {
        return a.request_id < b.request_id;
    });
    return batch;
}

void SearchEngine::submit_inference(
    const std::vector<std::uint64_t>& request_ids,
    const std::vector<std::vector<float>>& logits,
    const std::vector<float>& values) {
    if (request_ids.size() != logits.size() || request_ids.size() != values.size()) {
        throw std::runtime_error("inference batch shape mismatch");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (request_ids.empty()) return;

    std::vector<PendingEvaluation> pending_batch;
    pending_batch.reserve(request_ids.size());
    for (const auto request_id : request_ids) {
        const auto it = pending_.find(request_id);
        if (it == pending_.end()) throw std::runtime_error("unknown inference request id");
        pending_batch.push_back(it->second);
    }

    const std::size_t workers = std::max<std::size_t>(
        1, std::min<std::size_t>(request_ids.size(), static_cast<std::size_t>(config_.search_threads)));
    auto apply_range = [&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            const auto& pending = pending_batch[i];
            if (!pending.valid) continue;
            Node& node = tree_.node(pending.node_id);
            set_policy_and_value(node, logits[i], values[i], false);
        }
    };

    if (workers == 1 || request_ids.size() < 8) {
        apply_range(0, request_ids.size());
    } else {
        std::vector<std::thread> threads;
        threads.reserve(workers);
        const std::size_t chunk = (request_ids.size() + workers - 1) / workers;
        for (std::size_t worker = 0; worker < workers; ++worker) {
            const std::size_t begin = worker * chunk;
            const std::size_t end = std::min(request_ids.size(), begin + chunk);
            if (begin >= end) break;
            threads.emplace_back(apply_range, begin, end);
        }
        for (auto& thread : threads) thread.join();
    }

    for (std::size_t i = 0; i < request_ids.size(); ++i) {
        const auto& pending = pending_batch[i];
        Node& node = tree_.node(pending.node_id);
        if (!pending.valid) {
            node.pending = false;
            ++progress_epoch_;
        } else {
            ++progress_epoch_;
            backup(pending.path, values[i], false);
        }
        pending_.erase(request_ids[i]);
    }
}

std::vector<TrainingSample> SearchEngine::training_samples(std::size_t max_samples) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::int64_t> node_ids;
    node_ids.reserve(std::min(max_samples, tree_.size()));
    for (const Node& node : tree_.nodes()) {
        if (node_ids.size() >= max_samples) break;
        if (!node.initialized || node.terminal || node.pruned || node.closed ||
            node.edges.empty() || node.visits == 0) {
            continue;
        }
        const bool has_active = std::any_of(
            node.edges.begin(), node.edges.end(), [](const EdgeStats& edge) { return edge.active; });
        if (has_active) node_ids.push_back(node.id);
    }

    std::vector<TrainingSample> samples(node_ids.size());
    auto build_range = [&](std::size_t begin, std::size_t end) {
        for (std::size_t out_index = begin; out_index < end; ++out_index) {
            const Node& node = tree_.node(node_ids[out_index]);
            TrainingSample sample;
            sample.node_id = node.id;
            sample.global_features = global_features(node);
            sample.recent_residuals = recent_residual_features(node.id);
            sample.candidate_features = node.candidate_features;

            long double sum = 0.0L;
            const double inv_tau = 1.0 / std::max(1.0e-6, config_.visit_temperature);
            sample.policy_target.resize(node.edges.size(), 0.0f);
            std::size_t active = 0;
            for (std::size_t i = 0; i < node.edges.size(); ++i) {
                if (!node.edges[i].active) continue;
                ++active;
                const long double weight = std::pow(
                    static_cast<long double>(node.edges[i].visits), inv_tau);
                sample.policy_target[i] = static_cast<float>(weight);
                sum += weight;
            }
            if (sum > 0.0L) {
                for (auto& value : sample.policy_target) {
                    value = static_cast<float>(value / sum);
                }
            } else {
                const float uniform = 1.0f / static_cast<float>(active);
                for (std::size_t i = 0; i < node.edges.size(); ++i) {
                    sample.policy_target[i] = node.edges[i].active ? uniform : 0.0f;
                }
            }
            sample.has_value_target = node.has_terminal_score;
            sample.value_target = node.has_terminal_score
                ? static_cast<float>(node.best_terminal_score) : 0.0f;
            samples[out_index] = std::move(sample);
        }
    };

    const std::size_t workers = std::max<std::size_t>(
        1, std::min<std::size_t>(samples.size(), static_cast<std::size_t>(config_.search_threads)));
    if (workers == 1 || samples.size() < 256) {
        build_range(0, samples.size());
    } else {
        std::vector<std::thread> threads;
        threads.reserve(workers);
        const std::size_t chunk = (samples.size() + workers - 1) / workers;
        for (std::size_t worker = 0; worker < workers; ++worker) {
            const std::size_t begin = worker * chunk;
            const std::size_t end = std::min(samples.size(), begin + chunk);
            if (begin >= end) break;
            threads.emplace_back(build_range, begin, end);
        }
        for (auto& thread : threads) thread.join();
    }
    return samples;
}

std::vector<EvalRequest> SearchEngine::collect_refresh_batch(
    std::size_t cursor,
    std::size_t max_batch,
    std::size_t* next_cursor) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!pending_.empty()) throw std::runtime_error("global refresh requires zero pending inference requests");

    std::vector<std::int64_t> node_ids;
    node_ids.reserve(max_batch);
    std::size_t i = cursor;
    while (i < tree_.size() && node_ids.size() < max_batch) {
        const Node& node = tree_.node(static_cast<std::int64_t>(i));
        if (node.initialized && !node.terminal && !node.pruned && !node.closed &&
            !node.edges.empty()) {
            node_ids.push_back(node.id);
        }
        ++i;
    }
    if (next_cursor) *next_cursor = i;

    std::vector<EvalRequest> batch(node_ids.size());
    auto build_range = [&](std::size_t begin, std::size_t end) {
        for (std::size_t j = begin; j < end; ++j) {
            batch[j] = make_request(tree_.node(node_ids[j]), 0);
        }
    };
    const std::size_t workers = std::max<std::size_t>(
        1, std::min<std::size_t>(batch.size(), static_cast<std::size_t>(config_.search_threads)));
    if (workers == 1 || batch.size() < 64) {
        build_range(0, batch.size());
    } else {
        std::vector<std::thread> threads;
        threads.reserve(workers);
        const std::size_t chunk = (batch.size() + workers - 1) / workers;
        for (std::size_t worker = 0; worker < workers; ++worker) {
            const std::size_t begin = worker * chunk;
            const std::size_t end = std::min(batch.size(), begin + chunk);
            if (begin >= end) break;
            threads.emplace_back(build_range, begin, end);
        }
        for (auto& thread : threads) thread.join();
    }
    return batch;
}

void SearchEngine::apply_refresh(
    const std::vector<std::int64_t>& node_ids,
    const std::vector<std::vector<float>>& logits,
    const std::vector<float>& values) {
    if (node_ids.size() != logits.size() || node_ids.size() != values.size()) {
        throw std::runtime_error("refresh batch shape mismatch");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (!pending_.empty()) throw std::runtime_error("global refresh requires zero pending inference requests");

    auto apply_range = [&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            Node& node = tree_.node(node_ids[i]);
            set_policy_and_value(node, logits[i], values[i], false);
        }
    };
    const std::size_t workers = std::max<std::size_t>(
        1, std::min<std::size_t>(node_ids.size(), static_cast<std::size_t>(config_.search_threads)));
    if (workers == 1 || node_ids.size() < 64) {
        apply_range(0, node_ids.size());
    } else {
        std::vector<std::thread> threads;
        threads.reserve(workers);
        const std::size_t chunk = (node_ids.size() + workers - 1) / workers;
        for (std::size_t worker = 0; worker < workers; ++worker) {
            const std::size_t begin = worker * chunk;
            const std::size_t end = std::min(node_ids.size(), begin + chunk);
            if (begin >= end) break;
            threads.emplace_back(apply_range, begin, end);
        }
        for (auto& thread : threads) thread.join();
    }
    progress_epoch_ += static_cast<std::uint64_t>(node_ids.size());
}

bool SearchEngine::refresh_basis_with_best() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!pending_.empty()) throw std::runtime_error("basis refresh requires zero pending requests");
    if (refresh_candidate_z_.empty()) {
        throw std::runtime_error(
            "basis refresh requested but this phase found no non-basis terminal vector");
    }

    const auto result = lll_refresh(basis_, refresh_candidate_z_);
    if (!result.completed) {
        throw std::runtime_error("LLL refresh failed: " + result.error);
    }

    const std::string refreshed = result.basis.to_text();
    recorder_.add_phase(snapshot_phase(true, result.changed, refreshed));
    completed_phase_nodes_ += static_cast<std::uint64_t>(tree_.size());
    basis_ = result.basis;
    ++phase_index_;
    initialize_episode();
    return result.changed;
}

std::vector<float> SearchEngine::gso_features() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return gso_features_cache_;
}

std::vector<std::int64_t> SearchEngine::best_coefficients() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return best_z_;
}

std::vector<std::int64_t> SearchEngine::refresh_candidate_coefficients() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return refresh_candidate_z_;
}

std::vector<mpz_class> SearchEngine::best_vector() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return basis_.exact_vector(best_z_);
}

std::vector<PathRecord> SearchEngine::best_path_records() const {
    std::vector<PathRecord> records;
    const auto depth_counts = tree_.nodes_per_depth();

    if (best_terminal_node_id_ < 0) {
        const auto z = best_z_;
        long double rho = 0.0L;
        std::int64_t real_parent_id = tree_.root_id();
        const long double radius = radius_sq_scaled();
        for (int depth = 0; depth < basis_.dimension(); ++depth) {
            const int k = basis_.dimension() - 1 - depth;
            const auto geom = geometry_.legal_interval(k, rho, radius, z);
            const std::int64_t action = z[static_cast<std::size_t>(k)];
            const long double child_rho = geometry_.next_rho(
                k, rho, geom.center, action);

            PathRecord row;
            row.depth = static_cast<std::uint64_t>(depth + 1);
            row.coefficient_index = k;
            row.parent_id = real_parent_id;
            row.action = action;
            row.scale_exp2 = gso_.scale_exp2();
            row.nodes_at_depth = static_cast<std::size_t>(depth + 1) < depth_counts.size()
                ? depth_counts[static_cast<std::size_t>(depth + 1)] : 0;
            row.parent_rho = rho;
            row.rho = child_rho;
            row.incremental_cost = child_rho - rho;
            row.radius_sq = geom.radius_sq;
            row.remaining_sq = std::max(0.0L, geom.radius_sq - rho);
            row.center = geom.center;
            row.delta = geom.delta;
            row.legal_lo = geom.lo;
            row.legal_hi = geom.hi;
            row.normalized_offset = geom.delta > 0.0L
                ? static_cast<double>((static_cast<long double>(action) - geom.center) / geom.delta)
                : 0.0;

            std::int64_t next_real_parent = -1;
            if (real_parent_id >= 0 && static_cast<std::size_t>(real_parent_id) < tree_.size()) {
                const Node& real_parent = tree_.node(real_parent_id);
                const auto edge_it = std::find_if(
                    real_parent.edges.begin(), real_parent.edges.end(),
                    [&](const EdgeStats& edge) { return edge.action == action; });
                if (edge_it != real_parent.edges.end()) {
                    row.prior = edge_it->prior;
                    row.policy_prior = edge_it->policy_prior;
                    row.edge_visits = edge_it->visits;
                    row.q = edge_it->q;
                    row.m = edge_it->has_exact_m ? edge_it->m : edge_it->q;
                    if (edge_it->expanded && edge_it->child_id >= 0) {
                        row.node_id = edge_it->child_id;
                        next_real_parent = edge_it->child_id;
                    }
                }
            }
            records.push_back(row);
            rho = child_rho;
            real_parent_id = next_real_parent;
        }
        return records;
    }

    std::vector<std::int64_t> chain;
    std::int64_t current = best_terminal_node_id_;
    while (current > 0) {
        chain.push_back(current);
        current = tree_.node(current).parent_id;
    }
    std::reverse(chain.begin(), chain.end());

    for (std::size_t depth = 0; depth < chain.size(); ++depth) {
        const Node& child = tree_.node(chain[depth]);
        const Node& parent = tree_.node(child.parent_id);
        const auto edge_it = std::find_if(parent.edges.begin(), parent.edges.end(), [&](const EdgeStats& edge) {
            return edge.child_id == child.id;
        });
        PathRecord row;
        row.depth = depth + 1;
        row.coefficient_index = parent.next_k;
        row.node_id = child.id;
        row.parent_id = parent.id;
        row.action = child.action_from_parent;
        row.scale_exp2 = gso_.scale_exp2();
        row.nodes_at_depth = depth + 1 < depth_counts.size() ? depth_counts[depth + 1] : 0;
        row.parent_rho = parent.rho;
        row.rho = child.rho;
        row.incremental_cost = child.rho - parent.rho;
        row.radius_sq = parent.geometry.radius_sq;
        row.remaining_sq = std::max(0.0L, parent.geometry.radius_sq - parent.rho);
        row.center = parent.geometry.center;
        row.delta = parent.geometry.delta;
        row.legal_lo = parent.geometry.lo;
        row.legal_hi = parent.geometry.hi;
        row.normalized_offset = parent.geometry.delta > 0.0L
            ? static_cast<double>((static_cast<long double>(child.action_from_parent) - parent.geometry.center) / parent.geometry.delta)
            : 0.0;
        if (edge_it != parent.edges.end()) {
            row.prior = edge_it->prior;
            row.policy_prior = edge_it->policy_prior;
            row.edge_visits = edge_it->visits;
            row.q = edge_it->q;
            row.m = edge_it->has_exact_m ? edge_it->m : edge_it->q;
        }
        records.push_back(row);
    }
    return records;
}

PhaseSnapshot SearchEngine::snapshot_phase(
    bool basis_refresh_after,
    bool basis_refresh_changed,
    const std::string& refreshed_basis) const {
    PhaseSnapshot snapshot;
    snapshot.phase_index = phase_index_;
    snapshot.initial_basis_text = phase_initial_basis_;
    snapshot.total_nodes = tree_.size();
    snapshot.best_found_at_node_count = best_found_at_node_count_;
    snapshot.nodes_per_depth = tree_.nodes_per_depth();
    snapshot.initial_radius_squared = r0_sq_;
    snapshot.initial_b1_over_gh = input_quality_ratio_;
    snapshot.log_gh = gso_.log_gh();
    snapshot.gso_scale_exp2 = gso_.scale_exp2();
    snapshot.best_coefficients = best_z_;
    snapshot.best_vector = basis_.exact_vector(best_z_);
    snapshot.best_squared_norm = best_sq_;
    snapshot.best_score = best_score_;
    snapshot.best_path = best_path_snapshot_.empty() ? best_path_records() : best_path_snapshot_;
    snapshot.budget_reached = node_budget_reached();
    snapshot.tree_exhausted = tree_.node(tree_.root_id()).closed;
    snapshot.basis_refresh_after = basis_refresh_after;
    snapshot.basis_refresh_changed = basis_refresh_changed;
    snapshot.refreshed_basis_text = refreshed_basis;
    return snapshot;
}

void SearchEngine::write_results(
    const std::string& result_root,
    const std::string& version,
    const std::string& run_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    recorder_.add_phase(snapshot_phase(false, false, ""));
    recorder_.write(result_root, version, run_id);
}

}  // namespace mcts_enum
