cmake_minimum_required(VERSION 3.24)
project(mcts_enum_backend LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PYBIND11_RESULT
)
if(NOT PYBIND11_RESULT EQUAL 0)
    message(FATAL_ERROR "Cannot find pybind11 from ${Python3_EXECUTABLE}")
endif()
set(pybind11_DIR "${PYBIND11_CMAKE_DIR}" CACHE PATH "pybind11 CMake directory" FORCE)
find_package(pybind11 CONFIG REQUIRED)
find_package(Threads REQUIRED)

find_library(FPLLL_LIBRARY NAMES fplll REQUIRED)
find_library(GMP_LIBRARY NAMES gmp REQUIRED)
find_library(GMPXX_LIBRARY NAMES gmpxx REQUIRED)

pybind11_add_module(
    mcts_enum_backend
    NO_EXTRAS
    src/basis.cpp
    src/gso.cpp
    src/enumeration_geometry.cpp
    src/tree.cpp
    src/fplll_rollout.cpp
    src/search_engine.cpp
    src/result_recorder.cpp
    src/basis_refresh.cpp
    src/pybind_module.cpp
)

target_include_directories(mcts_enum_backend PRIVATE include)
target_compile_options(mcts_enum_backend PRIVATE -O3 -march=native)
target_link_libraries(
    mcts_enum_backend
    PRIVATE
        ${FPLLL_LIBRARY}
        ${GMPXX_LIBRARY}
        ${GMP_LIBRARY}
        Threads::Threads
)
