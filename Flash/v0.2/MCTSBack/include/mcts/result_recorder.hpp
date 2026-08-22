#pragma once

#include "mcts/types.hpp"

#include <string>
#include <vector>

namespace mcts_enum {

class ResultRecorder {
public:
    void set_search_config(const SearchConfig& config) {
        search_config_ = config;
        has_search_config_ = true;
    }
    void add_phase(PhaseSnapshot snapshot);
    void write(
        const std::string& result_root,
        const std::string& version,
        const std::string& run_id) const;

private:
    SearchConfig search_config_;
    bool has_search_config_ = false;
    std::vector<PhaseSnapshot> phases_;
};

}  // namespace mcts_enum
