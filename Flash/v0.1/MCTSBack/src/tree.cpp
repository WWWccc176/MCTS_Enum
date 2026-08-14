#include "mcts/tree.hpp"

#include <stdexcept>

namespace mcts_enum {

void Tree::reset(int dimension) {
    dimension_ = dimension;
    nodes_.clear();
    Node root;
    root.id = 0;
    root.parent_id = -1;
    root.next_k = dimension - 1;
    root.rho = 0.0L;
    root.terminal = dimension == 0;
    nodes_.push_back(std::move(root));
}

std::int64_t Tree::create_child(
    std::int64_t parent_id, std::size_t edge_index, long double rho) {
    const Node& parent_before = node(parent_id);
    if (edge_index >= parent_before.edges.size()) {
        throw std::runtime_error("invalid edge index");
    }
    const EdgeStats& edge_before = parent_before.edges[edge_index];
    if (edge_before.expanded) return edge_before.child_id;

    const std::int64_t child_id = static_cast<std::int64_t>(nodes_.size());
    const std::int64_t action = edge_before.action;
    const int next_k = parent_before.next_k - 1;

    Node child;
    child.id = child_id;
    child.parent_id = parent_id;
    child.action_from_parent = action;
    child.next_k = next_k;
    child.rho = rho;
    child.terminal = next_k < 0;
    nodes_.push_back(std::move(child));

    EdgeStats& committed_edge = node(parent_id).edges.at(edge_index);
    committed_edge.expanded = true;
    committed_edge.child_id = child_id;
    return child_id;
}

std::vector<std::uint64_t> Tree::nodes_per_depth() const {
    std::vector<std::uint64_t> counts(static_cast<std::size_t>(dimension_) + 1, 0);
    for (const Node& node : nodes_) {
        const int depth = dimension_ - 1 - node.next_k;
        if (depth >= 0 && depth <= dimension_) ++counts[static_cast<std::size_t>(depth)];
    }
    return counts;
}

}  // namespace mcts_enum
