#include "mcts/tree.hpp"

#include <algorithm>
#include <stdexcept>

namespace mcts_enum {

void Tree::reset(int dimension, std::size_t reserve_nodes) {
    dimension_ = dimension;
    nodes_.clear();
    if (reserve_nodes > 0) nodes_.reserve(reserve_nodes);

    Node root;
    root.id = 0;
    root.parent_id = -1;
    root.next_k = dimension - 1;
    nodes_.push_back(std::move(root));
}

std::int64_t Tree::create_child(
    std::int64_t parent_id,
    std::size_t edge_index,
    long double child_rho) {
    const Node& parent = node(parent_id);
    if (edge_index >= parent.edges.size()) throw std::runtime_error("invalid edge index");
    const EdgeStats& edge = parent.edges[edge_index];
    if (edge.expanded) return edge.child_id;

    Node child;
    child.id = static_cast<std::int64_t>(nodes_.size());
    child.parent_id = parent_id;
    child.next_k = parent.next_k - 1;
    child.action_from_parent = edge.action;
    child.rho = child_rho;
    child.prefix_high_to_low = parent.prefix_high_to_low;
    child.prefix_high_to_low.push_back(edge.action);

    const std::int64_t id = child.id;
    nodes_.push_back(std::move(child));
    EdgeStats& committed = node(parent_id).edges.at(edge_index);
    committed.expanded = true;
    committed.child_id = id;
    return id;
}

std::vector<std::uint64_t> Tree::nodes_per_depth() const {
    std::vector<std::uint64_t> counts(static_cast<std::size_t>(dimension_) + 1, 0);
    for (const Node& node : nodes_) {
        const std::size_t depth = node.prefix_high_to_low.size();
        if (depth < counts.size()) ++counts[depth];
    }
    return counts;
}

}  // namespace mcts_enum
