#!/bin/bash
#SBATCH --job-name=knapsack_shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=shared_memory.out
#SBATCH --error=shared_memory.err
#SBATCH --mem=32G
#SBATCH --exclusive

set -euo pipefail

# Superseded by run_timing.sbatch (standalone kpbench, pinning, repetitions,
# all solvers). Kept to reproduce the first Python-driven campaign.

# Pin the OpenMP threads (one per core, compact): without it threads migrate and
# the timings of a memory-bound kernel depend on where the scheduler puts them.
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"

NUM_THREADS=32
SEED=42
NUM_ITEMS=50
MAX_WEIGHT=5000
for max_weight in 10 100 1000; do
  for threads in 1 2 4 8 16 32; do
    for cap in 1000 10000 100000 1000000; do
      for item_block in 5 10 15 25; do
        for capacity_block in 10 100 1000 10000; do
          echo "========================================"
          echo "Capacity: $cap | Threads: $threads | MaxWeight: $max_weight| ItemBlock: $item_block | CapacityBlock: $capacity_block"
          echo "========================================"
          if [ "$capacity_block" -ge "$cap" ]; then
            continue
          fi
          echo "--- knapsackdpdag ---"
          python3 distributedknapsack \
            --test knapsackdpdag \
            --capacity "$cap" \
            --numItems "$NUM_ITEMS" \
            --seed "$SEED" \
            --maxWeight "$max_weight" \
            --itemBlock "$item_block" \
            --numThreads "$threads" \
            --capBlock "$capacity_block"
          echo ""
        done
      done
    done
  done
done
