#pragma once

#include "mcts/enumeration_geometry.hpp"
#include "mcts/types.hpp"

#include <cstdint>
#include <vector>

namespace mcts_enum {

struct Node {
    std::int64_t id = -1;
    std::int64_t parent_id = -1;
    int next_k = -1;
    std::int64_t action_from_parent = 0;
    long double rho = 0.0L;
    std::uint64_t visits = 0;
    bool initialized = false;
    bool geometry_ready = false;
    bool expanding = false;
    bool pending = false;
    bool terminal = false;
    bool pruned = false;
    bool closed = false;
    double value_estimate = 0.0;
    double best_terminal_score = -std::numeric_limits<double>::infinity();
    bool has_terminal_score = false;
    GeometryInfo geometry;
    std::vector<EdgeStats> edges;
    std::vector<float> candidate_features;
};

class Tree {
public:
    void reset(int dimension);
    std::int64_t create_child(std::int64_t parent_id, std::size_t edge_index,
                              long double rho);

    Node& node(std::int64_t id) { return nodes_.at(static_cast<std::size_t>(id)); }
    const Node& node(std::int64_t id) const { return nodes_.at(static_cast<std::size_t>(id)); }
    std::size_t size() const noexcept { return nodes_.size(); }
    std::int64_t root_id() const noexcept { return 0; }
    const std::vector<Node>& nodes() const noexcept { return nodes_; }
    std::vector<std::uint64_t> nodes_per_depth() const;

private:
    int dimension_ = 0;
    std::vector<Node> nodes_;
};

}  // namespace mcts_enum
