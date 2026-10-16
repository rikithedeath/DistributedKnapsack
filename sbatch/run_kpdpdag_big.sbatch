#!/bin/bash
#SBATCH --job-name=kpdpdag_big
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --mem=48G
#SBATCH --output=perf_big.out
#SBATCH --error=perf_big.err

# Scaling-crossover experiment: does the KnapsackDPDAG wall-clock scaling
# improve when the amount of WORK PER TILE grows?  The n=50 perf campaign
# (run_kpdpdag_perf.sbatch) showed S_wall saturating at ~2.5x with
# insn_ratio up to ~9x (spinning): the hypothesis is that synchronization
# dominates because each run is tiny.  Here the same capacity (1e6, table
# far above cache) is kept while num_items grows over {1000, 5000},
# i.e. 20x / 100x the per-tile work of the n=50 sweep (whose data already
# covers n=50):
#   - KnapsackDPDAG:  num_items {1000,5000} x itemBlock {10,25} x
#     capBlock {1000,10000} x threads {1,4,8,16,32}, maxWeight=10
#   - KnapsackDP baseline: num_items {1000,5000} x maxWeight {10}
#     x threads {1,4,8,16,32} (item_block/cap_block recorded as 0)
# 2 repeats each, same-run timing as run_kpdpdag_perf.sbatch.
#
# Runtime estimate (measured n=50 serial: DAG 0.16 s, plain 0.64 s at
# c=1e6; time scales ~linearly in num_items): ~6 min DAG + ~6 min plain
# + perf overhead -> ~13 min total.
# Memory: DP table is Eigen::Matrix<int> (4 B/cell): (5000+1)*(1e6+1)*4B
# ~= 20 GB plus graph overhead -> --mem=48G.
#
# Output: perf_big_out/perf_summary.csv, one row per (repeat x event),
# same convention as the other perf scripts plus two leading columns:
#       test,num_items,max_weight,capacity,threads,item_block,cap_block,tag,
#       value,unit,event,runtime_ns,runtime_pct[,metric_value,metric_unit],
#       repeat,time_s
# test distinguishes knapsackdpdag from the knapsackdp baseline; num_items
# is the sweep variable.  time_s = solver wall time of THAT repeat.

set -euo pipefail

SEED=42
CAPACITY=1000000
BIN="./build/kpbench"
REPEATS=2
OUT_DIR="perf_big_out"
SUMMARY="$OUT_DIR/perf_summary.csv"
SUMMARY_HEADER="test,num_items,max_weight,capacity,threads,item_block,cap_block,tag,value,unit,event,runtime_ns,runtime_pct,metric_value,metric_unit,repeat,time_s"
PERF_EVENTS="cycles,instructions,cache-references,cache-misses,branch-misses,L1-dcache-loads,L1-dcache-load-misses,ll_cache_rd,ll_cache_miss_rd,dTLB-load-misses,iTLB-load-misses"

export LC_ALL=C
mkdir -p "$OUT_DIR"
rm -f "$SUMMARY"

if [ ! -x "$BIN" ]; then
  echo "ERROR: $BIN not found or not executable." >&2
  echo "       Build it first: cmake --preset gpu && cmake --build build" >&2
  exit 1
fi

echo "perf_event_paranoid=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo n/a)"
if ! perf stat -e "$PERF_EVENTS" true >/dev/null 2>&1; then
  echo "ERROR: perf cannot use the configured PERF_EVENTS. Check perf_event_paranoid" >&2
  echo "       and event availability ('perf list'); aborting before the sweep." >&2
  exit 1
fi

run_one() { # $1=test $2=num_items $3=max_weight $4=threads $5=item_block $6=cap_block $7=rep
  local test="$1" ni="$2" mw="$3" threads="$4" ib="$5" cb="$6" rep="$7"
  local tag="n${ni}_${test}_maxw${mw}_cap${CAPACITY}_t${threads}_ib${ib}_cb${cb}"
  export OMP_NUM_THREADS="$threads"

  local extra_args=()
  if [ "$test" = "knapsackdpdag" ]; then
    extra_args=(--itemBlock "$ib" --capBlock "$cb")
  fi

  # perf writes its CSV to -o, so kpbench's stdout arrives here clean.
  local run_out
  run_out=$(perf stat -x , -e "$PERF_EVENTS" \
    -o "$OUT_DIR/perf_last.csv" \
    "$BIN" \
    --test "$test" \
    --capacity "$CAPACITY" \
    --numItems "$ni" \
    --seed "$SEED" \
    --numThreads "$threads" \
    --maxWeight "$mw" \
    "${extra_args[@]}" \
    --save "")

  # Solver wall time of THIS run, from kpbench's "<test>: <t>s | ..." line.
  local time_s
  time_s=$(printf '%s\n' "$run_out" | sed -n "s/^${test}: \([0-9.]*\)s .*/\1/p")
  if [ -z "$time_s" ]; then
    echo "WARNING: could not parse solver time for $tag rep $rep" >&2
  fi

  # Normalize each perf row to 7 fields, prefix run metadata, append;
  # the first fusion writes the CSV header as first line.
  [ -s "$SUMMARY" ] || echo "$SUMMARY_HEADER" > "$SUMMARY"
  awk -F, -v OFS=, -v test="$test" -v ni="$ni" -v mw="$mw" -v cap="$CAPACITY" \
    -v p="$threads" -v ib="$ib" -v cb="$cb" -v tag="$tag" \
    -v rep="$rep" -v t="$time_s" \
    '!/^#/ && NF >= 5 {
       mv = (NF >= 6) ? $6 : ""
       mu = (NF >= 7) ? $7 : ""
       print test, ni, mw, cap, p, ib, cb, tag, $1, $2, $3, $4, $5, mv, mu, rep, t
     }' \
    "$OUT_DIR/perf_last.csv" >>"$SUMMARY"
  rm -f "$OUT_DIR/perf_last.csv"
}

# --- KnapsackDPDAG: work-per-tile ladder ------------------------------------
for num_items in 1000 5000; do
  for threads in 1 4 8 16 32; do
    for item_block in 10 25; do
      for capacity_block in 1000 10000; do
        for rep in $(seq 1 "$REPEATS"); do
          run_one knapsackdpdag "$num_items" 10 "$threads" "$item_block" "$capacity_block" "$rep"
        done
      done
    done
  done
done

# --- KnapsackDP baseline ------------------------------------------------------
for num_items in 1000 5000; do
  for threads in 1 4 8 16 32; do
    for max_weight in 10; do
      for rep in $(seq 1 "$REPEATS"); do
        run_one knapsackdp "$num_items" "$max_weight" "$threads" 0 0 "$rep"
      done
    done
  done
done

echo "Outputs in $OUT_DIR/ (perf_summary.csv)"
echo "Analyze with:"
echo "  import pandas as pd"
echo "  df = pd.read_csv('$OUT_DIR/perf_summary.csv')"
echo "  # e.g. DAG wall time vs num_items at 16 threads:"
echo "  d = df[(df.test == 'knapsackdpdag') & (df.event == 'cycles')]"
echo "  d.groupby(['num_items', 'threads']).time_s.first()"
