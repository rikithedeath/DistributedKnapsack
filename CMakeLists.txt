cmake_minimum_required(VERSION 3.26)
project(DistributedKnapsack C CXX)


option(KP_GPU "Build and expose the CUDA/Thrust knapsack solver" OFF)
option(KP_MPI "Build and expose the MPI knapsack solver" OFF)

find_package(OpenMP REQUIRED)
find_package(fmt REQUIRED)
find_package(Python3 3.12 COMPONENTS Development Interpreter REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(spdlog REQUIRED)

if(KP_MPI)
  find_package(Boost REQUIRED COMPONENTS container mpi serialization)
  find_package(MPI REQUIRED)
else()
  find_package(Boost REQUIRED COMPONENTS container)
endif()
# Set C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# Collect all source files from src directory
file(GLOB_RECURSE KP_SRC "src/*.cpp")
# Keep the full source list for the benchmark executable
set(KP_BENCH_SRC ${KP_SRC})
# main.cpp must not end up in the shared library (it defines main())
list(REMOVE_ITEM KP_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")
# Collect module-like sources and expose them as CXX modules
file(GLOB_RECURSE KP_MODULES "src/*.cxx")
# Create executable
add_library(distributed_knapsack SHARED ${KP_SRC})
# Add module files as a CXX_MODULES file set (CMake 3.26+)
target_sources(distributed_knapsack PRIVATE
    FILE_SET CXX_MODULES
    TYPE CXX_MODULES
    FILES ${KP_MODULES}
)
if(ENABLE_TIMERS)
  target_compile_definitions(distributed_knapsack PUBLIC ENABLE_TIMERS=1)
endif()

# Include directories
target_include_directories(distributed_knapsack PUBLIC src include ${OpenMP_CXX_INCLUDE_DIRS} ${OpenMP_C_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${fmt_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ${spdlog_INCLUDE_DIRS})
target_compile_options(distributed_knapsack PUBLIC ${OpenMP_CXX_FLAGS} ${OpenMP_C_FLAGS} ${Boost_COMPILE_OPTIONS} )
# Link OpenMP libraries
target_link_libraries(distributed_knapsack PUBLIC OpenMP::OpenMP_CXX OpenMP::OpenMP_C fmt::fmt spdlog::spdlog pybind11::pybind11 Eigen3::Eigen)

if(KP_GPU)
  target_compile_definitions(distributed_knapsack PRIVATE KP_GPU=1)
  add_subdirectory(gpu)
  target_link_libraries(distributed_knapsack PUBLIC distributed_knapsack_gpu)
  target_include_directories(distributed_knapsack PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/gpu/include)
endif()

if(KP_MPI)
  target_compile_definitions(distributed_knapsack PRIVATE KP_MPI=1)
  add_subdirectory(mpi)
  target_link_libraries(distributed_knapsack PUBLIC distributed_knapsack_mpi)
  target_include_directories(distributed_knapsack PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/mpi ${CMAKE_CURRENT_SOURCE_DIR}/mpi/Knapsackmpi)
endif()

# Standalone benchmark executable (no pybind/Python symbols, no shared lib)
list(REMOVE_ITEM KP_BENCH_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/lib.cpp")
add_executable(kpbench ${KP_BENCH_SRC})
target_sources(kpbench PRIVATE
    FILE_SET CXX_MODULES
    TYPE CXX_MODULES
    FILES ${KP_MODULES}
)
target_include_directories(kpbench PRIVATE src external include ${OpenMP_CXX_INCLUDE_DIRS} ${OpenMP_C_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${fmt_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ${spdlog_INCLUDE_DIRS})
target_compile_options(kpbench PRIVATE ${OpenMP_CXX_FLAGS} ${OpenMP_C_FLAGS} ${Boost_COMPILE_OPTIONS})
target_link_libraries(kpbench PRIVATE OpenMP::OpenMP_CXX OpenMP::OpenMP_C fmt::fmt spdlog::spdlog Eigen3::Eigen)

if(ENABLE_TIMERS)
  target_compile_definitions(kpbench PRIVATE ENABLE_TIMERS=1)
endif()

if(KP_GPU)
  target_compile_definitions(kpbench PRIVATE KP_GPU=1)
  target_link_libraries(kpbench PRIVATE distributed_knapsack_gpu)
  target_include_directories(kpbench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/gpu/include)
endif()

add_subdirectory(test)
