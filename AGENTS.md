# DistributedKnapsack - Agent Instructions

## Project Overview

C++23 distributed knapsack solver using MPI, OpenMP, and Boost. Uses Conan for dependency management and CMake/Ninja for builds.

## Build Commands

**Configure and build (benchmarks: always Release):**
```bash
cmake --preset release && cmake --build build
```
`default` is a Debug build with the fine-grained timers on: development only.

**Run tests:**
```bash
./build/test/distributed_knapsack_tests
```

**Benchmark one run:**
```bash
./build/kpbench --test knapsackdp --numItems 1000 --capacity 1000000 --numThreads 8 --verify --save ""
```

**MPI tests (only with the `mpi` preset):**
```bash
mpirun -np 4 ./build/test/mpi/distributed_knapsack_mpi_tests
```

## Project Structure

- `src/` - library and benchmark sources
  - `main.cpp` - `kpbench`, the standalone benchmark (argparse)
  - `lib.cpp` - pybind11 module `libdistributed_knapsack`
  - `phases.hpp` - always-on coarse phase timers; `time.hpp` - fine-grained timers (`ENABLE_TIMERS`)
  - `build_info.hpp.in` - build provenance recorded in every result
  - `Knapsack/` - shared-memory solvers: `knapsackdp.cpp` (classic), `knapsackdpbits.cpp`
    (two rows + bit table), `knapsackdptiled.cpp` (tile wavefront), `knapsackdpdag.cpp`
    (tile DAG), `dp_kernel.hpp` (shared inner kernel and partitioning), COPA files
- `mpi/` - MPI solvers (`KP_MPI`), `gpu/` - CUDA solver (`KP_GPU`)
- `distributedknapsack/` - Python benchmark driver
- `sbatch/` - Slurm campaigns (`run_timing.sbatch` for timings, `run_*_perf.sbatch` for counters)
- `test/` - GTest unit tests
- `doc/` - report (`main.typ` + `chapters/`), analysis scripts, notes

## Dependencies (via Conan)

- `fmt`, `spdlog` - Logging/formatting
- `Boost::mpi`, `Boost::program_options`, `Boost::container`
- `OpenMP`, `MPI`, `Eigen3`, `GTest`

## Key Implementation Details

- every DP table is **row-major** (`Eigen::Matrix<..., Eigen::RowMajor>`): Eigen's default is
  column-major, which made consecutive cells `(n+1)*4` bytes apart;
- thread slices come from `kp::static_block` (half-open, 64-bit);
- tables are first-touched in parallel with the compute partition, timed as `first_touch`;
- all solvers except `knapsackdpbits` use `kp::dp_row_segment` as inner kernel;
- `knapsackdpdag` uses `solve_dag` (levels) by default, `solve_dag_topo` with `KNAPSACK_DAG_TOPO=1`.

**kpbench options:** `--test`, `--numItems`, `--capacity`, `--minWeight`, `--maxWeight`, `--minValue`,
`--maxValue`, `--seed`, `--numThreads`, `--itemBlock`, `--capBlock`, `--save` (JSON, empty disables),
`--verify`, `--dump-instance`.

## Testing Conventions

- Tests use GTest framework
- Test files: `test*.cpp` in `test/` and `test/mpi/`
- Test executable links against main target sources (excluding `main.cpp`)
- MPI tests require `mpirun`

## Editor/IDE Setup

- **clangd**: Uses `build/` as compilation database
- **Compiler**: Clang 20.x (`/usr/bin/clang`, `/usr/bin/clang++`)
- **CMake preset**: `default-clang` or `default-gcc`
- **Ignore**: `-fmodule*`, `-Wmodule*`, `-fdeps*` warnings

## Conan Workflow

Dependencies managed via `conanfile.py`. Conan generates:
- `build/conan_toolchain.cmake` - Toolchain file
- `build/CMakeUserPresets.json` - User presets

Run `conan install . --build=missing` before CMake configure if dependencies change.

## OpenCode Agent Config

Custom agents defined in `.opencode/agents/`:
- `docwriter.md` - Documentation agent (doc/ folder)
- `tester.md` - Testing agent (test/ folder)
