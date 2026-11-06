#!/bin/bash
#SBATCH --job-name=kpdpdag_perf
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=perf_kpdpdag.out
#SBATCH --error=perf_kpdpdag.err
#SBATCH --mem=32G
#SBATCH --exclusive

# Cache-miss investigation for KnapsackDPDAG with perf stat.
# Reduced sweep of run_kpdpdag.sbatch: max_weight {10,100,1000} x threads
# {1,2,4,8,16,32} x capacity {1e4,1e5,1e6} x itemBlock {5,10,15,25} x
# capBlock {1000,10000}, skipping capBlock >= capacity (360 runs), now with
# REPEATS repeats each.  Cache/TLB/branch events incl. the node's ll_cache
# last-level-cache counters, no perf record.  The target node is ARM
# (armv8_pmuv3 core PMU, arm_cmn mesh, arm_dsu_N L3 clusters, arm_dmc620
# memory controllers); LLC-loads/LLC-load-misses do not exist there.
# Extra core-PMU counters (l2d_cache_refill, l3d_cache_refill, stall_backend,
# mem_access, dtlb_walk, ...) or uncore ones (arm_cmn/hnf_cache_miss/,
# arm_dsu_0/l3d_cache_refill/, arm_dmc620_*/clkdiv2_rdwr/) can be added after
# checking `perf list`; uncore events count PMU instances, not the process,
# and unsupported events make perf fail.
# NOTE: run this on a node/account that is allowed to access hardware
# counters (perf_event_paranoid lowered or proper perms).
# capBlock=0 means "auto": (capacity+1)/10 (see knapsackdpdag.cpp); this
# sweep always passes explicit values so the tag is unambiguous.
#
# SAME-RUN TIMING: the measured process is the standalone ./build/kpbench
# binary (pure C++, no Python interpreter, no mpi4py/pybind imports), so perf
# counts ONLY the solver.  The solver wall time that kpbench prints on stdout
# ("knapsackdpdag: <t>s | ...") is captured per run and stored in the
# summary's time_s column, so counters and timings always belong to the SAME
# run.  This replaces the earlier workflow that joined counters with the
# single-shot times of doc/data/result_kpdpdag.json from a separate campaign
# (different runs and binary state, not comparable: the clock sanity check
# showed up to ~4x mismatches).  Build kpbench first:
#   cmake --preset release && cmake --build build
# Clean baseline timings for cross-checks remain in doc/data/result_kpdpdag.json.
#
# Machine-parseable output:
#   perf_kpdpdag_out/perf_summary.csv   all runs merged, one row per
#     (repeat x event), `perf stat -x ,` CSV rows prefixed with run metadata:
#       max_weight,capacity,threads,item_block,cap_block,tag,
#       value,unit,event,runtime_ns,runtime_pct[,metric_value,metric_unit],
#       repeat,time_s
#       runtime_ns = enabled time of the event's counter group, SUMMED over
#       the process's threads (not wall time);
#       runtime_pct = measurement coverage (100% = no multiplexing).
#       time_s = solver wall time (4 decimals) of THAT repeat; empty if the
#       result line could not be parsed.
#     unsupported events appear as value "<not supported>".  This node's perf
#     does not emit duration_time/user_time/system_time rows.
# Per-run perf files are NOT kept: each repeat's rows are appended to the
# summary and the temp file removed, so the summary is complete.
# LC_ALL=C forces '.' as decimal separator so CSV fields stay unambiguous.
# Walltime: worst runs are the serial cap=1e6 cap_block=1000 ones (~0.3 s x
# 3 repeats); with perf overhead the full sweep (360 x 3) fits comfortably
# in 2h.

set -euo pipefail

SEED=42
NUM_ITEMS=50
BIN="./build/kpbench"
REPEATS="${REPEATS:-3}"
TEST="knapsackdpdag"
OUT_DIR="perf_kpdpdag_out"
SUMMARY="$OUT_DIR/perf_summary.csv"
SUMMARY_HEADER="max_weight,capacity,threads,item_block,cap_block,tag,value,unit,event,runtime_ns,runtime_pct,metric_value,metric_unit,repeat,time_s"
PERF_EVENTS="cycles,instructions,cache-references,cache-misses,branch-misses,L1-dcache-loads,L1-dcache-load-misses,ll_cache_rd,ll_cache_miss_rd,dTLB-load-misses,iTLB-load-misses"

export LC_ALL=C

# Pin the OpenMP threads (one per core, compact): without it threads migrate and
# the timings of a memory-bound kernel depend on where the scheduler puts them.
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"
mkdir -p "$OUT_DIR"
rm -f "$SUMMARY"

if [ ! -x "$BIN" ]; then
  echo "ERROR: $BIN not found or not executable." >&2
  echo "       Build it first: cmake --preset release && cmake --build build" >&2
  exit 1
fi

# Refuse to measure a non-optimised binary (the "default" preset is Debug).
if ! "$BIN" --test bandwidth --capacity 1000 --save "" | grep -q '^build: Release'; then
  echo "ERROR: $BIN is not a Release build. Build it with:" >&2
  echo "       cmake --preset release && cmake --build build" >&2
  exit 1
fi

echo "perf_event_paranoid=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo n/a)"
if ! perf stat -e "$PERF_EVENTS" true >/dev/null 2>&1; then
  echo "ERROR: perf cannot use the configured PERF_EVENTS. Check perf_event_paranoid" >&2
  echo "       and event availability ('perf list'); aborting before the sweep." >&2
  exit 1
fi

for max_weight in 10 100 1000; do
  for threads in 1 2 4 8 16 32; do
    for cap in 10000 100000 1000000; do
      for item_block in 5 10 15 25; do
        for capacity_block in 1000 10000; do
          if [ "$capacity_block" -ge "$cap" ]; then
            continue
          fi
          tag="maxw${max_weight}_cap${cap}_t${threads}_ib${item_block}_cb${capacity_block}"
          export OMP_NUM_THREADS="$threads"

          for rep in $(seq 1 "$REPEATS"); do
            # perf writes its CSV to -o, so kpbench's stdout arrives here clean.
            RUN_OUT=$(perf stat -x , -e "$PERF_EVENTS" \
              -o "$OUT_DIR/perf_last.csv" \
              "$BIN" \
              --test "$TEST" \
              --capacity "$cap" \
              --numItems "$NUM_ITEMS" \
              --seed "$SEED" \
              --numThreads "$threads" \
              --maxWeight "$max_weight" \
              --itemBlock "$item_block" \
              --capBlock "$capacity_block" \
              --save "")

            # Solver wall time of THIS run, from kpbench's "<test>: <t>s | ..." line.
            time_s=$(printf '%s\n' "$RUN_OUT" | sed -n "s/^${TEST}: \([0-9.]*\)s .*/\1/p")
            if [ -z "$time_s" ]; then
              echo "WARNING: could not parse solver time for $tag rep $rep" >&2
            fi

            # Normalize each perf row to 7 fields, prefix run metadata, append;
            # the first fusion writes the CSV header as first line.
            [ -s "$SUMMARY" ] || echo "$SUMMARY_HEADER" > "$SUMMARY"
            awk -F, -v OFS=, -v mw="$max_weight" -v c="$cap" -v p="$threads" \
              -v ib="$item_block" -v cb="$capacity_block" -v tag="$tag" \
              -v rep="$rep" -v t="$time_s" \
              '!/^#/ && NF >= 5 {
                 mv = (NF >= 6) ? $6 : ""
                 mu = (NF >= 7) ? $7 : ""
                 print mw, c, p, ib, cb, tag, $1, $2, $3, $4, $5, mv, mu, rep, t
               }' \
              "$OUT_DIR/perf_last.csv" >>"$SUMMARY"
            rm -f "$OUT_DIR/perf_last.csv"
          done
        done
      done
    done
  done
done

echo "Outputs in $OUT_DIR/ (perf_summary.csv)"
echo "Each row carries repeat and time_s (solver wall time of that same run)."
echo "Analyze with:"
echo "  import pandas as pd"
echo "  df = pd.read_csv('$OUT_DIR/perf_summary.csv')"
echo "  # e.g. cache-miss counts per (capacity, item_block, cap_block) across threads:"
echo "  df[df.event == 'cache-misses'].pivot_table(index=['capacity', 'item_block', 'cap_block'], columns='threads', values='value')"
