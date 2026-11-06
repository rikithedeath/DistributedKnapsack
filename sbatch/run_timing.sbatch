#!/bin/bash
#SBATCH --job-name=kp_timing
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --mem=60G
#SBATCH --output=timing.out
#SBATCH --error=timing.err
#
# Time-only campaign for the CPU solvers: knapsackdp (classic, full row-major
# table), knapsackdpbits (classic, two rows + 1-bit decision table),
# knapsackdptiled (tiles on the full table, explicit anti-diagonal wavefront,
# no graph) and knapsackdpdag (tiles + Boost.Graph DAG, executed by levels).
#
# No perf here: perf stat with more events than hardware counters multiplexes
# and perturbs short runs. Hardware counters stay in run_*_perf.sbatch.
#
#   0. machine description saved next to the data (lscpu, NUMA, caches);
#   1. memory bandwidth (STREAM-like triad) for every thread count: the
#      ceiling of the roofline for these memory-bound kernels;
#   2. strong scaling: fixed problem, threads 1..80;
#   3. weak scaling: capacity proportional to the threads (work per thread
#      constant);
#   4. DAG structure: maxWeight below and above cap_block at fixed threads,
#      to test the "level(b,q) = b + q" assumption (maxWeight <= cap_block).
#
# Every configuration runs WARMUP discarded times, then REPEATS measured times;
# the first measured run also checks the optimum against an independent serial
# DP (--verify, not timed). Report medians with an interval (e.g. IQR or a
# bootstrap 95% CI) computed from the repeats.
#
# Each entry of $OUT/results.json carries: test, threads (effective), n, C,
# weight range and sum of weights (C >= sum_weights means "everything fits"),
# total time, per-phase times (alloc / first_touch / graph_build / compute /
# backtrack), build type, compiler, flags, git commit, OMP binding.
#
# Override with environment variables, e.g.
#   REPEATS=5 THREADS="1 2 4 8 16 32" sbatch sbatch/run_timing.sbatch
# Quick local smoke test of the whole script (capacities divided by 1000):
#   CAP_DIV=1000 REPEATS=1 WARMUP=0 THREADS="1 2" bash sbatch/run_timing.sbatch
# Build first:  cmake --preset release && cmake --build build
set -euo pipefail

BIN="./build/kpbench"
OUT="${OUT:-timing_out}"
RESULTS="$OUT/results.json"
LOG="$OUT/log.txt"
REPEATS="${REPEATS:-10}"
WARMUP="${WARMUP:-1}"
THREADS="${THREADS:-1 2 4 8 16 32 48 64 80}"
CAP_DIV="${CAP_DIV:-1}"   # >1 only for smoke tests: divides every capacity
SEED=42
CAP_BLOCK=$((10000 / CAP_DIV > 0 ? 10000 / CAP_DIV : 1))

export LC_ALL=C
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"

if [ ! -x "$BIN" ]; then
  echo "ERROR: $BIN not found. Build it first: cmake --preset release && cmake --build build" >&2
  exit 1
fi
if ! "$BIN" --test bandwidth --capacity 1000 --save "" | grep -q '^build: Release'; then
  echo "ERROR: $BIN is not a Release build: cmake --preset release && cmake --build build" >&2
  exit 1
fi

mkdir -p "$OUT"
rm -f "$RESULTS" "$LOG"

# 0. Machine description: the report takes cache sizes and topology from here.
{
  echo "date: $(date -Is)"; echo "host: $(hostname)"; echo "job: ${SLURM_JOB_ID:-none}"
  echo "--- lscpu"; lscpu
  echo "--- numactl -H"; numactl -H 2>/dev/null || echo "numactl not available"
  echo "--- caches (cpu0)"
  for d in /sys/devices/system/cpu/cpu0/cache/index*; do
    echo "L$(cat "$d/level") $(cat "$d/type") $(cat "$d/size") shared_cpu_list=$(cat "$d/shared_cpu_list")"
  done
  echo "--- kpbench build"; "$BIN" --test bandwidth --capacity 1000 --save "" | head -1
} > "$OUT/machine.txt" 2>&1

# run <threads> <kpbench args...>
run() {
  local p="$1"; shift
  export OMP_NUM_THREADS="$p"
  local r
  for ((r = 0; r < WARMUP; r++)); do
    "$BIN" "$@" --numThreads "$p" --save "" > /dev/null
  done
  for ((r = 1; r <= REPEATS; r++)); do
    local extra=()
    [ "$r" -eq 1 ] && [ "${2:-}" != "bandwidth" ] && extra=(--verify)
    echo "### rep=$r threads=$p $*" >> "$LOG"
    # A wrong result must not abort the campaign: it is logged and checked at the end.
    "$BIN" "$@" --numThreads "$p" --save "$RESULTS" ${extra[@]+"${extra[@]}"} >> "$LOG" \
      || echo "ERROR: exit status $? for: threads=$p $*" >> "$LOG"
  done
}

# solvers <threads> <n> <capacity> <max_weight>
solvers() {
  local p="$1" n="$2" c="$(( $3 / CAP_DIV ))" mw="$4"
  local common=(--numItems "$n" --capacity "$c" --maxWeight "$mw" --seed "$SEED")
  run "$p" --test knapsackdp "${common[@]}"
  run "$p" --test knapsackdpbits "${common[@]}"
  local ib
  for ib in 5 10 25; do
    run "$p" --test knapsackdptiled "${common[@]}" --itemBlock "$ib" --capBlock "$CAP_BLOCK"
    run "$p" --test knapsackdpdag "${common[@]}" --itemBlock "$ib" --capBlock "$CAP_BLOCK"
  done
}

# 1. Bandwidth: 3 arrays of 1e8 doubles (2.4 GB), far above any cache.
for p in $THREADS; do
  run "$p" --test bandwidth --capacity $((100000000 / CAP_DIV))
done

# 2. Strong scaling. (n, C) pairs: the table is (n+1)(C+1) ints, up to 20 GB.
for nc in "50 100000" "50 1000000" "50 10000000" "1000 100000" "1000 1000000" "5000 100000" "5000 1000000"; do
  read -r n c <<< "$nc"
  for mw in 10 1000; do
    for p in $THREADS; do
      solvers "$p" "$n" "$c" "$mw"
    done
  done
done

# 3. Weak scaling: n = 1000, C = 50000 * p (16 GB table at 80 threads).
for p in $THREADS; do
  solvers "$p" 1000 $((50000 * p)) 1000
done

# 4. DAG structure vs maxWeight / cap_block (n = 1000, C = 1e6, 32 threads).
for mw in 10 1000 10000 100000; do
  for cb in 1000 10000; do
    for test in knapsackdptiled knapsackdpdag; do
      run "${STRUCT_THREADS:-32}" --test "$test" --numItems 1000 --capacity $((1000000 / CAP_DIV)) \
        --maxWeight "$mw" --seed "$SEED" --itemBlock 10 --capBlock "$(( cb / CAP_DIV > 0 ? cb / CAP_DIV : 1 ))"
    done
  done
done

if grep -q "MISMATCH\|ERROR" "$LOG"; then
  echo "ERROR: at least one run returned a wrong or invalid solution, see $LOG" >&2
  exit 2
fi
echo "Done: $RESULTS ($(grep -c '"testname"' "$RESULTS") entries), machine description in $OUT/machine.txt"
