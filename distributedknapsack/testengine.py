import sys
import time
import random
import argparse
import json
import os
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional
from pathlib import Path
try:
    # Only the MPI solver needs mpi4py; shared-memory runs must not depend on it
    # (importing mpi4py also initialises MPI).
    from mpi4py import MPI  # type: ignore
except ImportError:  # pragma: no cover - depends on the environment
    MPI = None
from libdistributed_knapsack import (KnapsackArguments, KnapsackSolution, knapsackdp, knapsackdpbits,
                                     knapsackdptiled, knapsackcopa, knapsackcopasequential, knapsackdpdag,
                                     knapsackdpmpi, knapsackdpgpu, get_all_sections, get_dag_stats,
                                     get_phases, omp_max_threads, build_info)


def ensure_mpi_initialized() -> None:
    if MPI is None:
        raise RuntimeError("mpi4py is not installed: the MPI solver is unavailable")
    if not MPI.Is_initialized():
        MPI.Init()


def _env_int(name: str, fallback: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return fallback


def resolve_processors(test_name: str, test: "BenchmarkTest") -> int:
    """Return the effective number of processors used by a test.

    MPI tests use the communicator size, the GPU kernel uses KP_GPU_THREADS
    while the OpenMP based CPU tests use OMP_NUM_THREADS.
    """
    if test.is_mpi_test:
        return MPI.COMM_WORLD.size
    if "gpu" in test_name:
        return _env_int("KP_GPU_THREADS", test.numThreads)
    # What OpenMP really uses, not the environment variable: setting
    # OMP_NUM_THREADS after libgomp has been loaded has no effect.
    return omp_max_threads()


class BenchmarkTest(ABC):
    def __init__(self) -> None:
        self.args: KnapsackArguments
        self.numThreads: int = 1
        self.numItems: int = 0
        self.testType :str = ""
        self.is_mpi_test: bool = False
        self.is_dag_test: bool = False
        self.is_tiled_test: bool = False
    @abstractmethod
    def onExecute(self) -> KnapsackSolution:
        pass

    def setup(self, weights: list[int], values: list[int], 
              capacity: int, numThreads: int = 1) -> None:
        self.args =    KnapsackArguments(weights, values, capacity)
        self.numThreads = numThreads
        self.numItems = len(weights)

    def execute(self) -> Tuple[float, KnapsackSolution]:
        start_time = time.perf_counter()  # monotonic, unlike time.time()
        result = self.onExecute()
        end_time = time.perf_counter()
        return end_time - start_time, result



class BenchmarkKnapsackDP(BenchmarkTest):
    def onExecute(self) -> KnapsackSolution:
        return knapsackdp(self.args)


class BenchmarkKnapsackDPBits(BenchmarkTest):
    def onExecute(self) -> KnapsackSolution:
        return knapsackdpbits(self.args)


class BenchmarkKnapsackDPTiled(BenchmarkTest):
    def __init__(self, item_block: int = 10, cap_block: int = 0) -> None:
        super().__init__()
        self.item_block = item_block
        self.cap_block = cap_block
        self.is_tiled_test = True

    def onExecute(self) -> KnapsackSolution:
        return knapsackdptiled(self.args, self.item_block, self.cap_block)


class BenchmarkKnapsackCOPA(BenchmarkTest):
    def onExecute(self) -> KnapsackSolution:
        return knapsackcopa(self.args)


class BenchmarkKnapsackCOPASerial(BenchmarkTest):
    def onExecute(self) -> KnapsackSolution:
        return knapsackcopasequential(self.args)


class BenchmarkKnapsackDPDAG(BenchmarkTest):
    def __init__(self, item_block: int = 10, cap_block: int = 0) -> None:
        super().__init__()
        self.item_block = item_block
        self.cap_block = cap_block
        self.is_dag_test = True
        self.is_tiled_test = True

    def onExecute(self) -> KnapsackSolution:
        return knapsackdpdag(self.args, self.item_block, self.cap_block)


class BenchmarkKnapsackDPMPI(BenchmarkTest):
    def __init__(self) -> None:
        super().__init__()
        self.is_mpi_test = True

    def onExecute(self) -> KnapsackSolution:
        ensure_mpi_initialized()
        return knapsackdpmpi(self.args)


class BenchmarkKnapsackDPGPU(BenchmarkTest):
    def onExecute(self) -> KnapsackSolution:
        return knapsackdpgpu(self.args)


class TestRegister:
    def __init__(self, save_file: Optional[str] = None, capacity: int = 0,
                 min_weight: int = 0, max_weight: int = 0, seed: int = 0) -> None:
        self._tests: Dict[str, BenchmarkTest] = {}
        self._save_file = save_file
        self._capacity = capacity
        self._min_weight = min_weight
        self._max_weight = max_weight
        self._seed = seed

    def register(self, name: str, test: BenchmarkTest) -> None:
        self._tests[name] = test

    def setup(self, weights: list[int], values: list[int], numThreads: int = 1) -> None:
        """Setup all registered tests with the same data."""
        for test in self._tests.values():
            test.setup(weights, values, self._capacity, numThreads)

    def _append_result(self, test_name: str, test: BenchmarkTest,
                       duration: float, result: KnapsackSolution,
                       sections: dict, stats, phases) -> None:
        """Append test result (with time sections and DAG frontier stats) to JSON file."""
        if not self._save_file:
            return

        processors = resolve_processors(test_name, test)
        test_type = "distributed memory" if test.is_mpi_test else "shared memory"
        hostname = MPI.Get_processor_name() if test.is_mpi_test else os.uname().nodename
        build = build_info()

        entry = {
            'hostname': hostname,
            'testname': test_name,
            'testtype': test_type,
            'time': float(duration),
            'processors': processors,
            'solution_weight': result.totalWeight,
            'solution_profit': result.totalValue,
            'capacity': self._capacity,
            'num_items': test.numItems,
            'min_weight': self._min_weight,
            'max_weight': self._max_weight,
            'seed': self._seed,
            'time_sections': {
                section_name: {
                    'count': section.count,
                    'mean': section.mean,
                    'min': section.min,
                    'max': section.max,
                    'variance': section.variance,
                }
                for section_name, section in sections.items()
            },
            'phases': {name: seconds for name, seconds in phases},
            'build': dict(build),
            'omp': {
                'threads_effective': omp_max_threads(),
                'proc_bind': os.environ.get('OMP_PROC_BIND', ''),
                'places': os.environ.get('OMP_PLACES', ''),
            },
        }

        if test.is_tiled_test:
            entry['item_block'] = getattr(test, 'item_block', None)
            entry['cap_block'] = getattr(test, 'cap_block', None)

        if test.is_dag_test and stats.levels > 0:
            entry['frontier_distribution'] = {
                'min': stats.frontierMin,
                'median': stats.frontierMedian,
                'mean': stats.frontierMean,
                'max': stats.frontierMax,
            }
            entry['dag'] = {
                'tiles': stats.tiles,
                'edges': stats.edges,
                'levels': stats.levels,
            }

        entries = []
        if os.path.exists(self._save_file) and os.path.getsize(self._save_file) > 0:
            try:
                with open(self._save_file) as f:
                    entries = json.load(f)
            except (json.JSONDecodeError, ValueError):
                entries = []

        entries.append(entry)

        with open(self._save_file, 'w') as f:
            json.dump(entries, f, indent=2)

    def run(self, name: str = "all") -> None:
        if name == "all":
            tests_to_run = self._tests
        else:
            if name not in self._tests:
                raise ValueError(f"Test '{name}' not found")
            tests_to_run = {name: self._tests[name]}
        
        for test_name, test in tests_to_run.items():
            duration, result = test.execute()

            if test.is_mpi_test and MPI.COMM_WORLD.rank != 0:
                continue  # Only rank 0 prints/saves results for MPI tests

            print(f"{test_name}: {duration:.4f}s | Items: {test.numItems} | "
                  f"Profit: {result.totalValue} | Weight: {result.totalWeight}")

            sections = get_all_sections()
            if sections:
                print("  Time sections:")
                for name, sec in sorted(sections.items()):
                    print(f"    {name}: count={sec.count} mean={sec.mean:.4f}ms "
                          f"min={sec.min:.4f}ms max={sec.max:.4f}ms")

            phases = get_phases()
            if phases:
                print("  phases: " + " ".join(f"{n}={t:.4f}s" for n, t in phases))

            # DAG statistics are global "last DAG run" state: only meaningful
            # right after a DAG run (they would be stale for any other test).
            stats = get_dag_stats() if test.is_dag_test else None
            if stats:
                print(f"    DAG stats: tiles={stats.tiles} edges={stats.edges} "
                      f"levels={stats.levels}")
                if stats.levels > 0:
                    print(f"    wavefront widths: min={stats.frontierMin:.0f} "
                          f"median={stats.frontierMedian:.1f} mean={stats.frontierMean:.2f} "
                          f"max={stats.frontierMax:.0f}")

            if self._save_file:
                self._append_result(test_name, test, duration, result, sections, stats, phases)



