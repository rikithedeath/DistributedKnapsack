#!/bin/bash
#SBATCH --job-name=knapsack_gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=gpu.out
#SBATCH --error=gpu.err
#SBATCH --mem=32G

set -euo pipefail

SEED=42
NUM_ITEMS=50
BIN="./build/kpbench"
REPEATS=3
TEST="knapsackdp_gpu"
OUT_DIR="gpu_out"
SUMMARY="$OUT_DIR/gpu_summary.csv"

mkdir -p "$OUT_DIR"
echo "max_weight,capacity,threads,repeat,time_s,dp_ms" > "$SUMMARY"

for max_weight in 10 1000; do
  for threads in 32 64 128 256 512 1024; do
    for cap in 100000 1000000 10000000; do
      echo "========================================"
      echo "Capacity: $cap | GPU threads/block: $threads | maxWeight: $max_weight"
      echo "========================================"

      export KP_GPU_THREADS="$threads"

      for rep in $(seq 1 "$REPEATS"); do
        RUN_OUT=$("$BIN" \
          --test "$TEST" \
          --capacity "$cap" \
          --numItems "$NUM_ITEMS" \
          --seed "$SEED" \
          --maxWeight "$max_weight" \
          --save "")

        # total wall time from kpbench stdout: "knapsackdp_gpu: 1.2345s | ..."
        time_s=$(printf '%s\n' "$RUN_OUT" | sed -n "s/^${TEST}: \([0-9.]*\)s .*/\1/p")
        # kernel-only time: "GPU DP execution: 12.345 ms"
        dp_ms=$(printf '%s\n' "$RUN_OUT" | sed -n "s/^GPU DP execution: \([0-9.]*\) ms/\1/p")

        echo "$max_weight,$cap,$threads,$rep,$time_s,$dp_ms" >> "$SUMMARY"
        echo "rep=$rep time=${time_s}s dp=${dp_ms}ms"
      done
      echo ""
    done
  done
done

echo "Done. Summary written to $SUMMARY"
