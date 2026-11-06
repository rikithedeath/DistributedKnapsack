import random
import os
import sys
from pathlib import Path
from typing import Tuple
BUILD_DIR = Path(__file__).parent.parent / "build"
sys.path.insert(0, str(BUILD_DIR))

import argparse

def generate_data(num_items: int, min_weight: int, max_weight: int,
                  min_value: int, max_value: int, seed: int) -> Tuple[list[int], list[int]]:
    random.seed(seed)
    weights = [random.randint(min_weight, max_weight) for _ in range(num_items)]
    values = [random.randint(min_value, max_value) for _ in range(num_items)]
    return weights, values


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Distributed Knapsack Benchmark")
    parser.add_argument("--test", type=str, default="all", 
                        help="Test to run: knapsackdp, knapsackdpbits, knapsackdptiled, knapsackcopa, "
                             "knapsackcopa_serial, knapsackdpdag, knapsackdp_mpi, knapsackdp_gpu, all")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--numItems", type=int, default=30, help="Number of items")
    parser.add_argument("--numThreads", type=int, default=1, help="Number of threads")
    parser.add_argument("--capacity", type=int, default=1000, help="Knapsack capacity")
    parser.add_argument("--minWeight", type=int, default=1, help="Minimum weight")
    parser.add_argument("--maxWeight", type=int, default=50, help="Maximum weight")
    parser.add_argument("--minValue", type=int, default=1, help="Minimum value")
    parser.add_argument("--maxValue", type=int, default=50, help="Maximum value")
    parser.add_argument("--save", type=str, default="results.json", 
                        help="Output JSON file for results (default: results.json)")
    parser.add_argument("--itemBlock", type=int, default=10, help="Item block size for knapsackdpdag (default: 10)")
    parser.add_argument("--capBlock", type=int, default=0, help="Capacity block size for knapsackdpdag (default: 0)")
    
    args = parser.parse_args()
    # OMP_NUM_THREADS is read once, when libgomp is loaded together with the
    # extension module: it must be set BEFORE the import below. (It used to be
    # set after the import and had no effect: every run used the default team.)
    os.environ["OMP_NUM_THREADS"] = str(args.numThreads)

    import libdistributed_knapsack as lib
    from testengine import (
        TestRegister, BenchmarkKnapsackDP, BenchmarkKnapsackDPBits, BenchmarkKnapsackDPTiled,
        BenchmarkKnapsackCOPA, BenchmarkKnapsackCOPASerial, BenchmarkKnapsackDPDAG,
        BenchmarkKnapsackDPMPI, BenchmarkKnapsackDPGPU
    )

    lib.set_num_threads(args.numThreads)
    if lib.omp_max_threads() != args.numThreads:
        print(f"WARNING: requested {args.numThreads} threads, OpenMP will use {lib.omp_max_threads()}",
              file=sys.stderr)
    
    weights, values = generate_data(
        args.numItems, args.minWeight, args.maxWeight,
        args.minValue, args.maxValue, args.seed
    )
    
    register = TestRegister(save_file=args.save, capacity=args.capacity,
                            min_weight=args.minWeight, max_weight=args.maxWeight,
                            seed=args.seed)
    register.register("knapsackdp", BenchmarkKnapsackDP())
    register.register("knapsackdpbits", BenchmarkKnapsackDPBits())
    register.register("knapsackdptiled", BenchmarkKnapsackDPTiled(args.itemBlock, args.capBlock))
    register.register("knapsackcopa", BenchmarkKnapsackCOPA())
    register.register("knapsackcopa_serial", BenchmarkKnapsackCOPASerial())
    register.register("knapsackdpdag", BenchmarkKnapsackDPDAG(args.itemBlock, args.capBlock))
    register.register("knapsackdp_mpi", BenchmarkKnapsackDPMPI())
    register.register("knapsackdp_gpu", BenchmarkKnapsackDPGPU())
    
    register.setup(weights, values, args.numThreads)
    register.run(args.test)
