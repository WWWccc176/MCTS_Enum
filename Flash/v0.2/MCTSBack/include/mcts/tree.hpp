#pragma once

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
    std::uint32_t in_flight = 0;
    bool pending_rollout = false;
    bool closed = false;
    bool geometry_ready = false;
    std::uint64_t geometry_epoch = 0;

    GeometryInfo geometry;
    std::vector<std::int64_t> prefix_high_to_low;
    std::vector<EdgeStats> edges;

    bool se_initialized = false;
    bool se_exhausted = false;
    bool se_left_done = false;
    bool se_right_done = false;
    std::int64_t se_left = 0;
    std::int64_t se_right = 1;
};

class Tree {
public:
    void reset(int dimension, std::size_t reserve_nodes = 0);
    std::int64_t create_child(
        std::int64_t parent_id,
        std::size_t edge_index,
        long double child_rho);

    Node& node(std::int64_t id) { return nodes_.at(static_cast<std::size_t>(id)); }
    const Node& node(std::int64_t id) const { return nodes_.at(static_cast<std::size_t>(id)); }
    std::int64_t root_id() const noexcept { return 0; }
    std::size_t size() const noexcept { return nodes_.size(); }
    const std::vector<Node>& nodes() const noexcept { return nodes_; }
    std::vector<std::uint64_t> nodes_per_depth() const;

private:
    int dimension_ = 0;
    std::vector<Node> nodes_;
};

}  // namespace mcts_enum
