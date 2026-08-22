#include "mcts/search_engine.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <string>

namespace py = pybind11;
using namespace mcts_enum;

PYBIND11_MODULE(mcts_enum_backend, module) {
    module.doc() =
        "Flash v0.2 hybrid MCTS + fplll Schnorr-Euchner subtree enumeration backend";

    py::class_<SearchConfig>(module, "SearchConfig")
        .def(py::init<>())
        .def_readwrite("node_budget", &SearchConfig::node_budget)
        .def_readwrite("unlimited_nodes", &SearchConfig::unlimited_nodes)
        .def_readwrite("search_threads", &SearchConfig::search_threads)
        .def_readwrite("rollout_dimensions", &SearchConfig::rollout_dimensions)
        .def_readwrite("rollout_solutions", &SearchConfig::rollout_solutions)
        .def_readwrite("w_m", &SearchConfig::w_m)
        .def_readwrite("lambda_puct", &SearchConfig::lambda_puct)
        .def_readwrite("cpw", &SearchConfig::cpw)
        .def_readwrite("dpw", &SearchConfig::dpw)
        .def_readwrite("numeric_guard_rel", &SearchConfig::numeric_guard_rel)
        .def_readwrite("numeric_guard_abs", &SearchConfig::numeric_guard_abs)
        .def_readwrite("lll_delta", &SearchConfig::lll_delta)
        .def_readwrite("refresh_potential_rel_tolerance", &SearchConfig::refresh_potential_rel_tolerance);

    py::class_<SearchEngine>(module, "SearchEngine")
        .def(py::init([](py::bytes packet, const SearchConfig& config, bool nn_enabled) {
            const std::string bytes = packet;
            return std::make_unique<SearchEngine>(Basis::from_packet(bytes), config, nn_enabled);
        }), py::arg("basis_packet"), py::arg("config"), py::arg("nn_enabled") = false)
        .def_property_readonly("dimension", &SearchEngine::dimension)
        .def_property_readonly("node_count", &SearchEngine::node_count)
        .def_property_readonly("work_node_count", &SearchEngine::work_node_count)
        .def_property_readonly("input_quality_ratio", &SearchEngine::input_quality_ratio)
        .def_property_readonly("finished", &SearchEngine::finished)
        .def("diagnostic_status", &SearchEngine::diagnostic_status)
        .def("status", [](const SearchEngine& engine) {
            const auto s = engine.status_snapshot();
            py::dict out;
            out["phase"] = s.phase;
            out["tree_nodes"] = s.tree_nodes;
            out["enumeration_nodes"] = s.enumeration_nodes;
            out["work_nodes"] = s.work_nodes;
            out["node_budget"] = s.unlimited_nodes ? -1LL : static_cast<long long>(s.node_budget);
            out["rollout_jobs"] = s.rollout_jobs;
            out["exact_candidates"] = s.exact_candidates;
            out["best_found_at_work_node_count"] = s.best_found_at_work_node_count;
            out["refresh_candidate_found_at_work_node_count"] = s.refresh_candidate_found_at_work_node_count;
            out["root_visits"] = s.root_visits;
            out["radius_drops"] = s.radius_drops;
            out["initial_quality_ratio"] = s.initial_quality_ratio;
            out["best_quality_ratio"] = s.best_quality_ratio;
            out["search_radius_quality_ratio"] = s.search_radius_quality_ratio;
            out["refresh_candidate_quality_ratio"] = s.refresh_candidate_quality_ratio;
            out["refresh_candidate_available"] = s.refresh_candidate_available;
            out["budget_reached"] = s.budget_reached;
            out["root_closed"] = s.root_closed;
            out["active_rollouts"] = s.active_rollouts;
            return out;
        })
        .def("run_flash", &SearchEngine::run_flash, py::call_guard<py::gil_scoped_release>())
        .def("refresh_basis_with_best", &SearchEngine::refresh_basis_with_best,
             py::call_guard<py::gil_scoped_release>())
        .def("refresh_candidate_coefficients", &SearchEngine::refresh_candidate_coefficients)
        .def("best_vector", [](const SearchEngine& engine) {
            py::list out;
            for (const auto& x : engine.best_vector()) out.append(x.get_str());
            return out;
        })
        .def("current_basis_text", &SearchEngine::current_basis_text)
        .def("current_basis_packet", [](const SearchEngine& engine) {
            return py::bytes(engine.current_basis_packet());
        })
        .def("write_results", &SearchEngine::write_results,
             py::arg("result_root"), py::arg("version"), py::arg("run_id"));

    module.def("encode_basis_text", [](const std::string& text) {
        return py::bytes(Basis::from_text(text).to_packet());
    });
    module.attr("BACKEND_VERSION") = kBackendVersion;
}
