"""Analysis of the perf-stat hardware counters in doc/data/perf/.

Three CSVs produced by sbatch/run_kpdp_perf.sbatch,
sbatch/run_kpdpdag_perf.sbatch and sbatch/run_kpdpdag_big.sbatch (ARM node,
armv8_pmuv3 core PMU) are analysed:

    perf_kpdp.csv          KnapsackDP, 12 runs: max_weight {10, 1000} x
                           capacity {1e6, 1e7} x threads {1, 8, 32}
    perf_kpdpdag.csv       KnapsackDPDAG (tiled DP over a dependency DAG),
                           360 runs: max_weight {10, 100, 1000} x capacity
                           {1e4, 1e5, 1e6} x threads {1..32} x item_block
                           {5, 10, 15, 25} x cap_block {1000, 10000}
                           (cap_block >= capacity skipped)
    perf_kpdpbig.csv       scaling-crossover sweep at capacity=1e6
                           (run_kpdpdag_big.sbatch): KnapsackDPDAG,
                           num_items {1000, 5000} x item_block {10, 25} x
                           cap_block {1000, 10000} x threads {1..32}
                           (max_weight=10), plus a KnapsackDP baseline
                           num_items {1000, 5000} x max_weight {10}; rows
                           carry explicit test/num_items columns

The measured process is the standalone ./build/kpbench binary, so the
counters cover only the solver (the older *_python.csv files in the same
folder come from python3-distributedknapsack wrappings that additionally
count interpreter startup + imports; they are out of scope here).

Every CSV row is one `perf stat -x ,` event of one run.  The 11 events are
pivoted to columns and all ratios are recomputed from the raw counter
values; the derived metric_value columns of perf are used only as a
cross-check (they are rounded to 2 decimals).

The sbatch scripts capture the solver wall time on the stdout of the SAME
run that perf measures (REPEATS=3 replicates per configuration), stored in
the trailing CSV columns `repeat` and `time_s`.  Every timing therefore
comes from the perf campaign itself: no JSON join, no cross-campaign
mixing (the legacy doc/data/result_*.json timings come from different
runs and binaries and are no longer used here).

Both files are aggregated per configuration with the MEDIAN over the three
replicates (counters and time alike); the per-replicate rows are kept for
the repeatability and reliability diagnostics.  Nothing is dropped or
filtered: coverage and consistency checks are REPORT-ONLY.

Notation (w = max_weight family, c = capacity, p = threads/processors):

    ev(w, c, p)              raw counter value of event ev
    IPC(p)   = instructions / cycles                instructions per cycle
    MPKI_e(p) = miss_e / instructions * 1000        misses per kilo-instruction,
                e in {L1 (L1-dcache-load-misses), LLC (ll_cache_miss_rd), dTLB}
    hit_e(p)  = 1 - miss_e / loads_e                cache hit ratio (%)
    B(p)      = branch-misses / instructions * 100  branch-miss rate (%)
    cycles_per_cell = cycles / ((num_items + 1) * (c + 1))
                                                cycles per DP table cell; the
                                                table is the same for both
                                                algorithms, so K_c is the
                                                hardware work per cell that
                                                makes them comparable (for
                                                p > 1 it counts the cycles of
                                                all cores, i.e. it also grows
                                                with spinning and overhead)
    S_wall(p) = T(1) / T(p)                         wall speedup (same-run
                                                    median times of the
                                                    perf campaign)
    S_cyc(p)  = cycles(1) / cycles(p)               cycle-count reduction; note
                                                  that perf counts cycles summed
                                                  over ALL cores, so S_cyc < 1
                                                  when p > 1 cores run
    rate(p)   = cycles(p) / (T(p) * p)              cycles per core-second
                                                  (perf cycles, same-run T)
    r(p)      = rate(p) / rate(1)                   per-core cycle-rate ratio
                                                  (core frequency x busy share)
    identity:  S_wall(p) = S_cyc(p) * p * r(p)
    tiles     = ceil(num_items/item_block) * ceil((c+1)/cap_block)
                                                number of DAG tiles
    coverage  = min over events of runtime_pct      PMU multiplexing quality
    insn_ratio(p) = instructions(p) / instructions(1)
                                                total instruction count vs the
                                                serial run of the same
                                                configuration (<1 means the
                                                parallel runs issue fewer
                                                instructions in total, i.e.
                                                work redistribution or
                                                multiplexing noise)
    clock(p)  = cycles(p) / (T(p) * p)              implied per-core clock from
                                                    perf cycles and the same-run
                                                    median wall time; ~2-3.5 GHz
                                                    on these ARM nodes, values
                                                    outside that band flag a
                                                    counter inconsistency
                                                    (REPORT-ONLY)
    bw_est(p) = (llc_read_misses * 64B
                 + (n+1)*(c+1)*4B) / T(p)          estimated DRAM bandwidth
                                                    (GB/s): read traffic from
                                                    the last-level read misses
                                                    (lower bound: prefetch hits
                                                    escape the counter), write
                                                    traffic modeled as one 4 B
                                                    write per DP cell
                                                    (REPORT-ONLY estimate)

Counter speedups always compare the same configuration at p = 1.  For
KnapsackDPDAG the per-p best (item_block, cap_block) configuration is
selected by median wall time; its serial baseline is the best configuration
at p = 1.

Coverage is REPORT-ONLY: no run is dropped, the minimum coverage per run is
carried in the tables and summarised once per file.

Plots are written to doc/images/ with a "perf_" prefix.

Run with:  ~/pyenv/sci/bin/python doc/kpdp_perf_analysis.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PERF_DIR = BASE_DIR / "data" / "perf"
KPDP_PERF_FILE = PERF_DIR / "perf_kpdp.csv"
DAG_PERF_FILE = PERF_DIR / "perf_kpdpdag.csv"
BIG_PERF_FILE = PERF_DIR / "perf_kpdpbig.csv"
OUT_DIR = BASE_DIR / "images"

NUM_ITEMS = 50  # every benchmark run uses num_items=50 (see sbatch scripts)
REPEATS = 3     # replicates per configuration captured by the sbatch scripts

FAMILY_ORDER = [10, 100, 1000]
FAMILY_COLORS = {10: "#1f77b4", 100: "#ff7f0e", 1000: "#2ca02c"}
CAP_COLORS = {
    10_000: "#1f77b4",
    100_000: "#ff7f0e",
    1_000_000: "#2ca02c",
    10_000_000: "#d62728",
}
BLOCK_COLORS = {1000: "#1f77b4", 10000: "#d62728"}
IB_MARKERS = {5: "o", 10: "s", 15: "^", 25: "D", 50: "v"}
N_COLORS = {1000: "#1f77b4", 5000: "#2ca02c"}
THREAD_COLORS = {1: "#1f77b4", 8: "#ff7f0e", 32: "#d62728"}

EVENTS = (
    "cycles",
    "instructions",
    "cache-references",
    "cache-misses",
    "branch-misses",
    "L1-dcache-loads",
    "L1-dcache-load-misses",
    "ll_cache_rd",
    "ll_cache_miss_rd",
    "dTLB-load-misses",
    "iTLB-load-misses",
)

# Reference point for the (item_block x cap_block) deep-dive.
BLOCK_GRID_POINT = {"family": 10, "capacity": 100_000, "processors": 8}
# Intersection grid for the cross-algorithm comparison.
CMP_CAP = 1_000_000
CMP_FAMILIES = [10, 1000]
CMP_THREADS = [1, 8, 32]


def _key_cols(columns: pd.Index) -> list[str]:
    """Run-identity columns present in a perf CSV.

    The crossover sweep CSV additionally carries `test` (knapsackdp /
    knapsackdpdag) and `num_items`; the n=50 campaign CSVs get a constant
    num_items column injected by the loader.
    """
    cols = []
    if "test" in columns:
        cols.append("test")
    if "num_items" in columns:
        cols.append("num_items")
    cols += ["max_weight", "capacity", "threads"]
    if "item_block" in columns:
        cols += ["item_block", "cap_block"]
    return cols


def load_perf_csv(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pivot one perf summary CSV (one row per run x event) to wide form.

    The sbatch scripts append the same-run solver wall time (`time_s`) and
    the replicate index (`repeat`, REPEATS=3) to every row.  The wide frame
    therefore has one row per (run, replicate).

    Returns (wide, long):
        wide   one row per run x replicate, one column per event, plus
               time_s, runtime_ns (longest-scheduled counter group) and
               coverage_pct (minimum runtime_pct over the replicate's
               events, i.e. worst multiplexing)
        long   the raw CSV rows (used for the metric cross-check)
    """
    long = pd.read_csv(path)
    if "repeat" not in long.columns or "time_s" not in long.columns:
        raise ValueError(
            f"{path.name}: missing 'repeat'/'time_s' columns -- this is an "
            "old-campaign CSV without same-run timings"
        )
    if "num_items" not in long.columns:
        long.insert(0, "num_items", NUM_ITEMS)  # constant in the n=50 sweeps
    keys = _key_cols(long.columns)
    for col in keys + ["repeat"]:
        if col != "test":
            long[col] = long[col].astype(int)
    long["value_num"] = pd.to_numeric(long["value"], errors="coerce")
    long["time_s"] = pd.to_numeric(long["time_s"], errors="coerce")
    bad = long["value_num"].isna().sum() + long["time_s"].isna().sum()
    if bad:
        print(f"WARNING: {path.name}: {bad} non-numeric counter/time values dropped")

    wide = long.pivot(index=keys + ["repeat"], columns="event", values="value_num").reset_index()
    times = long.groupby(keys + ["repeat"], as_index=False)["time_s"].first()
    sched = (
        long.groupby(keys + ["repeat"])
        .agg(runtime_ns=("runtime_ns", "max"), coverage_pct=("runtime_pct", "min"))
        .reset_index()
    )
    wide = wide.merge(times, on=keys + ["repeat"]).merge(sched, on=keys + ["repeat"])
    return wide, long


def aggregate_repeats(wide: pd.DataFrame) -> pd.DataFrame:
    """Collapse the REPEATS=3 replicates of each run to one row.

    Every numeric quantity (counters and same-run time alike) is summarised
    by its median over the replicates.  Repeatability diagnostics are kept:
        time_spread   max(time_s)/min(time_s) across replicates
        coverage_pct  worst PMU coverage across replicates
    Nothing is dropped or filtered; the per-replicate rows stay available
    for the reliability tables.
    """
    keys = _key_cols(wide.columns)
    agg = wide.groupby(keys, as_index=False).agg(
        **{event: (event, "median") for event in EVENTS},
        time=("time_s", "median"),
        time_min=("time_s", "min"),
        time_max=("time_s", "max"),
        runtime_ns=("runtime_ns", "median"),
        coverage_pct=("coverage_pct", "min"),
    )
    agg["time_spread"] = agg["time_max"] / agg["time_min"]
    return agg


def crosscheck_metrics(long: pd.DataFrame, name: str) -> None:
    """Compare perf's own derived metric_value columns with recomputed ratios.

    perf attaches derived metrics to the event they are computed from:
        insn per cycle              -> instructions / cycles
        of all cache refs           -> cache-misses / cache-references * 100
        of all L1-dcache accesses   -> L1-misses / L1-loads * 100
    perf rounds them to two decimals, so a 1% tolerance is applied.
    """
    denom_of = {
        "insn per cycle": "cycles",
        "of all cache refs": "cache-references",
        "of all L1-dcache accesses": "L1-dcache-loads",
    }
    scale_of = {"insn per cycle": 1.0, "of all cache refs": 100.0, "of all L1-dcache accesses": 100.0}
    keys = _key_cols(long.columns) + ["repeat"]

    rows = long[long["metric_value"].notna() & long["metric_unit"].isin(denom_of)].copy()
    if rows.empty:
        print(f"{name}: no perf-derived metric rows to cross-check")
        return
    dens = long[long["event"].isin(denom_of.values())][keys + ["event", "value_num"]].rename(
        columns={"event": "den_event", "value_num": "den_value"}
    )
    rows["den_event"] = rows["metric_unit"].map(denom_of)
    rows = rows.merge(dens, on=keys + ["den_event"], how="left")
    expected = rows["metric_unit"].map(scale_of) * rows["value_num"] / rows["den_value"]
    bad = ((expected - rows["metric_value"]).abs() > 0.01 * rows["metric_value"].abs() + 0.005).sum()
    print(f"{name}: cross-checked {len(rows)} perf-derived metrics, {bad} mismatches beyond rounding")


def add_counter_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the shared counter metrics (same definitions for both files).

        IPC    = instructions / cycles
        MPKI_e = miss_e / instructions * 1000      for L1, LLC and dTLB misses
        hit_e  = (1 - miss_e / loads_e) * 100
        B      = branch-misses / instructions * 100
        K_c    = cycles / ((num_items + 1) * (capacity + 1))
    """
    df = df.copy()
    df["ipc"] = df["instructions"] / df["cycles"]
    df["mpki_l1"] = df["L1-dcache-load-misses"] / df["instructions"] * 1000.0
    df["mpki_llc"] = df["ll_cache_miss_rd"] / df["instructions"] * 1000.0
    df["l1_hit_pct"] = (1.0 - df["L1-dcache-load-misses"] / df["L1-dcache-loads"]) * 100.0
    df["llc_hit_pct"] = (1.0 - df["ll_cache_miss_rd"] / df["ll_cache_rd"]) * 100.0
    df["mpki_dtlb"] = df["dTLB-load-misses"] / df["instructions"] * 1000.0
    df["br_miss_pct"] = df["branch-misses"] / df["instructions"] * 100.0
    df["cycles_per_cell"] = df["cycles"] / ((df["num_items"] + 1) * (df["capacity"] + 1))
    return df.replace([np.inf, -np.inf], np.nan)


LINE_BYTES = 64  # cache-line size assumed by the DRAM traffic estimate


def add_bandwidth(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the estimated DRAM traffic and bandwidth (REPORT-ONLY estimate).

        dram_read_gb  = ll_cache_miss_rd * 64 B   every last-level read miss
                        must fetch one line from memory; hardware-prefetch
                        hits escape the counter, so this is a LOWER bound
        dram_write_gb = (num_items+1) * (capacity+1) * 4 B   every DP cell is
                        written exactly once and, at these table sizes, the
                        line is evicted before the cell is read again
                        (write-back traffic model)
        bw_gbs        = (dram_read_gb + dram_write_gb) / time

    Totals of the run divided by the same-run median wall time.  Used to
    check whether the parallel runs saturate the node's DRAM bandwidth
    (flat bw_gbs vs threads) or keep growing (compute/latency bound).
    """
    df = df.copy()
    cells = (df["num_items"] + 1) * (df["capacity"] + 1)
    df["dram_read_gb"] = df["ll_cache_miss_rd"] * LINE_BYTES / 1e9
    df["dram_write_gb"] = cells * 4.0 / 1e9
    df["bw_gbs"] = (df["dram_read_gb"] + df["dram_write_gb"]) / df["time"]
    return df.replace([np.inf, -np.inf], np.nan)


def add_insn_ratio(df: pd.DataFrame, on: list[str], proc_col: str = "processors") -> pd.DataFrame:
    """Attach insn_ratio = instructions(p) / instructions(1) of the same setup.

    The baseline is the p = 1 run sharing the `on` keys (same (family,
    capacity) for KnapsackDP, additionally the same tile configuration for
    KnapsackDPDAG).  The same total table should need roughly the same total
    instructions, so a ratio clearly below 1 at large p is unexpected: it is
    either work redistribution or multiplexing noise (coverage is only
    ~50% here).  REPORT-ONLY diagnostic, never used to filter runs.
    """
    df = df.copy()
    serial = df[df[proc_col] == 1].set_index(on)["instructions"].rename("insns_base")
    df = df.merge(serial.reset_index(), on=on, how="left", validate="many_to_one")
    df["insn_ratio"] = df["instructions"] / df["insns_base"]
    return df.replace([np.inf, -np.inf], np.nan)


def print_clock_check(df: pd.DataFrame, name: str) -> None:
    """Report-only consistency check: implied per-core clock = cycles/(T*p).

    perf counts cycles summed over all cores, so dividing by the same-run
    median wall time and the thread count gives the average per-core clock.
    On these ARM nodes that must land in roughly [1, 3.5] GHz; values above
    the band mean the counters disagree with the timing (too-short runs,
    coverage-scaling artifacts), values below it mean heavy time spent
    outside the measured compute (spinning, I/O).
    """
    clock = df["cycles"] / (df["time"] * df["processors"]) / 1e9
    out = pd.DataFrame({
        "clock_ghz": clock,
        "processors": df["processors"],
        "time_s": df["time"],
        "coverage_pct": df["coverage_pct"],
    })
    bad_hi = clock > 3.5
    bad_lo = clock < 1.0
    print(
        f"\n{name}: implied per-core clock GHz: min={clock.min():.2f} "
        f"median={clock.median():.2f} max={clock.max():.2f} "
        f"({int(bad_hi.sum())} runs above 3.5, {int(bad_lo.sum())} below 1.0)"
    )
    worst = out[bad_hi | bad_lo].sort_values("clock_ghz", ascending=False)
    if len(worst):
        print(worst.round(3).head(12).to_string(index=False))


def report_coverage(wide: pd.DataFrame, name: str) -> None:
    """Coverage summary (report-only, nothing is filtered)."""
    cov = wide["coverage_pct"]
    low = (cov < 60.0).sum()
    print(
        f"{name}: coverage min={cov.min():.0f}% median={cov.median():.0f}% "
        f"({low}/{len(cov)} replicate runs below 60%)"
    )


def print_repeatability(wide: pd.DataFrame, name: str) -> None:
    """Run-to-run repeatability of the same-run solver times (REPORT-ONLY).

    The sbatch scripts repeat every configuration REPEATS=3 times and parse
    the solver wall time from kpbench's stdout on each replicate; this
    quantifies how stable those wall times are.
    """
    keys = _key_cols(wide.columns)
    spread = wide.groupby(keys)["time_s"].agg(lambda s: s.max() / s.min())
    print(
        f"{name}: replicate time spread (max/min over {REPEATS} replicates): "
        f"min={spread.min():.3f} median={spread.median():.3f} max={spread.max():.3f} "
        f"({int((spread > 1.1).sum())}/{len(spread)} configurations above 1.1)"
    )


def build_kpdp(perf: pd.DataFrame) -> pd.DataFrame:
    """Aggregate KnapsackDP replicates and add the speedup split.

        time            median same-run solver wall time over the replicates
        speedup         = time(p=1) / time(p), computed within the campaign
                          per (max_weight, capacity)
        s_cycles        = cycles(1) / cycles(p)   same family and capacity
        rate_ratio      = rate(p) / rate(1), rate = cycles / (T * p)
    """
    merged = aggregate_repeats(perf)
    merged = add_counter_metrics(merged)
    merged = add_bandwidth(merged)

    serial = merged[merged["threads"] == 1].set_index(["num_items", "max_weight", "capacity"])[
        ["time", "cycles", "ipc"]
    ].rename(columns={"time": "t_serial", "cycles": "cycles_base", "ipc": "ipc_base"})
    merged = merged.merge(serial.reset_index(), on=["num_items", "max_weight", "capacity"])
    merged["speedup"] = merged["t_serial"] / merged["time"]
    merged = add_insn_ratio(merged, ["num_items", "max_weight", "capacity"], proc_col="threads")
    merged["s_cycles"] = merged["cycles_base"] / merged["cycles"]
    merged["s_ipc"] = merged["ipc"] / merged["ipc_base"]
    merged["rate_ratio"] = (merged["cycles"] / (merged["time"] * merged["threads"])) / (
        merged["cycles_base"] / merged["t_serial"]
    )
    merged = merged.rename(columns={"max_weight": "family", "threads": "processors"})
    return merged


def build_dag(perf: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate DAG replicates and attach same-run timings.

    Returns (configs, best):
        configs      every (item_block, cap_block) configuration: median
                     counters + median same-run wall time + repeatability
        best         per (family, capacity, processors) the fastest
                     configuration by median wall time, with S_wall /
                     S_cyc / rate_ratio against the best configuration at
                     p = 1 of the same (family, capacity)
    """
    configs = aggregate_repeats(perf).rename(
        columns={"max_weight": "family", "threads": "processors"}
    )
    configs = add_counter_metrics(configs)
    configs = add_bandwidth(configs)
    configs = add_insn_ratio(
        configs, ["num_items", "family", "capacity", "item_block", "cap_block"]
    )
    configs["tiles"] = (
        np.ceil(configs["num_items"] / configs["item_block"])
        * np.ceil((configs["capacity"] + 1) / configs["cap_block"])
    )

    # Per-p best configuration by median same-run wall time.
    idx = configs.groupby(["num_items", "family", "capacity", "processors"])["time"].idxmin()
    best = configs.loc[idx].reset_index(drop=True)

    # Serial baseline: best configuration at p=1 of the same (family, capacity).
    serial = best[best["processors"] == 1].set_index(["num_items", "family", "capacity"])
    best = best.merge(
        serial[["time", "item_block", "cap_block", "cycles"]].rename(
            columns={
                "time": "t_base",
                "item_block": "ib_base",
                "cap_block": "cb_base",
                "cycles": "cycles_base",
            }
        ),
        on=["num_items", "family", "capacity"],
    )
    best["s_wall"] = best["t_base"] / best["time"]
    best["s_cycles"] = best["cycles_base"] / best["cycles"]
    best["rate_ratio"] = (best["cycles"] / (best["time"] * best["processors"])) / (
        best["cycles_base"] / best["t_base"]
    )
    return configs, best


def _style_axis_threads(ax, processors: list[int]) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(processors)
    ax.set_xticklabels([str(p) for p in processors])
    ax.grid(True, which="both", alpha=0.3)


def print_kpdp_tables(merged: pd.DataFrame) -> None:
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    cols = [
        "family", "capacity", "processors", "time", "time_spread", "speedup", "ipc", "s_cycles",
        "rate_ratio", "mpki_l1", "mpki_llc", "mpki_dtlb", "l1_hit_pct", "llc_hit_pct",
        "br_miss_pct", "cycles_per_cell", "insn_ratio", "bw_gbs", "coverage_pct",
    ]
    print("\n=== KnapsackDP counters (same-run median timings, 3 replicates) ===")
    print(merged[cols].sort_values(["family", "capacity", "processors"]).round(3).to_string(index=False))

    top = merged[merged["processors"] == merged["processors"].max()]
    print("\n=== KnapsackDP speedup decomposition at max threads ===")
    print(
        top[["family", "capacity", "speedup", "s_cycles", "s_ipc", "rate_ratio", "ipc",
             "mpki_llc", "mpki_dtlb", "insn_ratio"]]
        .rename(columns={"speedup": "s_wall"})
        .round(4)
        .to_string(index=False)
    )


def print_dag_tables(configs: pd.DataFrame, best: pd.DataFrame) -> None:
    point = BLOCK_GRID_POINT
    sub = configs[
        (configs["family"] == point["family"])
        & (configs["capacity"] == point["capacity"])
        & (configs["processors"] == point["processors"])
    ]
    print(
        f"\n=== KnapsackDPDAG block grid (max_weight={point['family']}, "
        f"capacity={point['capacity']:,}, threads={point['processors']}) ==="
    )
    for label, col in (("time (s)", "time"), ("IPC", "ipc"), ("LLC MPKI", "mpki_llc")):
        print(f"--- {label} ---")
        print(
            sub.pivot(index="item_block", columns="cap_block", values=col)
            .sort_index(ascending=False)
            .round(4)
            .to_string()
        )

    cols = [
        "family", "capacity", "processors", "item_block", "cap_block", "time", "time_spread",
        "s_wall", "s_cycles", "rate_ratio", "ipc", "mpki_l1", "mpki_llc", "mpki_dtlb",
        "br_miss_pct", "insn_ratio", "tiles", "coverage_pct",
    ]
    print("\n=== KnapsackDPDAG best configuration per (family, capacity, threads) ===")
    print(best[cols].sort_values(["family", "capacity", "processors"]).round(4).to_string(index=False))

    best_map = best.assign(
        cfg=best["item_block"].astype(str) + "/" + best["cap_block"].astype(str)
    )
    print("\n=== KnapsackDPDAG fastest (item_block/cap_block) per (family, capacity, threads) ===")
    print(
        best_map.pivot_table(index=["family", "capacity"], columns="processors",
                             values="cfg", aggfunc="first")
        .to_string()
    )


def print_dag_reliability(configs: pd.DataFrame, best: pd.DataFrame) -> None:
    """How trustworthy are the DAG wall times?  (REPORT-ONLY)

    The wall times are now the same-run solver times of the perf campaign
    (median over three highly repeatable replicates), but most runs still
    finish in tens of milliseconds where PMU multiplexing (coverage down to
    <20%) degrades the scaled counters.  Diagnostics:
      1. duration buckets vs coverage,
      2. replicate spread (max/min time) of the configurations,
      3. best/worst configuration time ratio per (family, capacity, threads)
         and the best-config wall time vs threads (should grow the work, not
         stay flat).
    """
    print("\n=== KnapsackDPDAG timing reliability ===")
    print(f"runs: {len(configs)}, {int((configs['time'] < 0.1).sum())} shorter than 100 ms")
    for lo, hi in ((0.0, 0.05), (0.05, 0.1), (0.1, 1.0)):
        sel = configs[(configs["time"] >= lo) & (configs["time"] < hi)]
        if len(sel):
            print(
                f"  {lo * 1e3:.0f}-{hi * 1e3:.0f} ms: {len(sel):3d} runs, "
                f"coverage min={sel['coverage_pct'].min():.0f}% "
                f"median={sel['coverage_pct'].median():.0f}%"
            )

    spread = configs["time_spread"]
    print(
        "replicate spread max/min time per configuration: "
        f"min={spread.min():.3f} median={spread.median():.3f} max={spread.max():.3f} "
        f"({int((spread > 1.1).sum())}/{len(spread)} configs above 1.1)"
    )

    cfg_spread = configs.groupby(["family", "capacity", "processors"])["time"].agg(
        lambda s: s.max() / s.min()
    )
    print(
        "best/worst config time ratio per (family, capacity, threads): "
        f"min={cfg_spread.min():.2f} median={cfg_spread.median():.2f} max={cfg_spread.max():.2f}"
    )

    times = best.assign(t_ms=best["time"] * 1e3).pivot_table(
        index=["family", "capacity"], columns="processors", values="t_ms"
    )
    print("\nbest-config wall time (ms) vs threads (flat rows = no visible scaling):")
    print(times.round(1).to_string())


def print_comparison_table(kpdp: pd.DataFrame, best: pd.DataFrame) -> None:
    """Cross-algorithm comparison on the common grid (same perf campaign).

    Both sweeps ran in the same perf campaign on epito01 and both algorithms
    fill the same (num_items+1) x (capacity+1) DP table, so the same-run
    wall times, IPC, K_c and MPKI are directly comparable without any
    cross-campaign join.  The total LLC-miss counts and instruction counts
    make the headline effect explicit: tiling cuts the absolute number of
    LLC misses (llc_reduction = misses_kpdp / misses_dag) while issuing a
    similar number of instructions.
    """
    rows = []
    for family in CMP_FAMILIES:
        for proc in CMP_THREADS:
            k = kpdp[(kpdp["family"] == family) & (kpdp["capacity"] == CMP_CAP)
                     & (kpdp["processors"] == proc)]
            d = best[(best["family"] == family) & (best["capacity"] == CMP_CAP)
                     & (best["processors"] == proc)]
            if k.empty or d.empty:
                print(f"WARNING: comparison grid incomplete for family={family}, p={proc}")
                continue
            k, d = k.iloc[0], d.iloc[0]
            rows.append({
                "family": family,
                "processors": proc,
                "t_kpdp": k["time"],
                "t_dag": d["time"],
                "time_ratio_dag/kpdp": d["time"] / k["time"],
                "dag_config": f"ib{int(d['item_block'])}_cb{int(d['cap_block'])}",
                "insns_kpdp": k["instructions"],
                "insns_dag": d["instructions"],
                "ipc_kpdp": k["ipc"],
                "ipc_dag": d["ipc"],
                "kcell_kpdp": k["cycles_per_cell"],
                "kcell_dag": d["cycles_per_cell"],
                "llc_miss_kpdp": k["ll_cache_miss_rd"],
                "llc_miss_dag": d["ll_cache_miss_rd"],
                "llc_reduction": k["ll_cache_miss_rd"] / d["ll_cache_miss_rd"],
                "mpki_llc_kpdp": k["mpki_llc"],
                "mpki_llc_dag": d["mpki_llc"],
            })
    table = pd.DataFrame(rows)
    print("\n=== Cross-algorithm comparison (capacity=1e6, same perf campaign) ===")
    print(table.round(4).to_string(index=False))


def build_big_plain(wide: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the KnapsackDP baseline of the crossover sweep.

    Per (num_items, max_weight, processors): median counters + same-run
    median time, wall speedup against p = 1 of the same (num_items,
    max_weight), and the DRAM bandwidth estimate.
    """
    merged = aggregate_repeats(wide[wide["test"] == "knapsackdp"])
    merged = add_counter_metrics(merged)
    merged = add_bandwidth(merged)
    serial = merged[merged["threads"] == 1].set_index(["num_items", "max_weight"])[
        ["time", "cycles"]
    ].rename(columns={"time": "t_serial", "cycles": "cycles_base"})
    merged = merged.merge(serial.reset_index(), on=["num_items", "max_weight"])
    merged["speedup"] = merged["t_serial"] / merged["time"]
    merged["s_cycles"] = merged["cycles_base"] / merged["cycles"]
    merged = add_insn_ratio(merged, ["num_items", "max_weight"], proc_col="threads")
    merged["rate_ratio"] = (merged["cycles"] / (merged["time"] * merged["threads"])) / (
        merged["cycles_base"] / merged["t_serial"]
    )
    return merged.rename(columns={"threads": "processors"})


def print_big_tables(best_big: pd.DataFrame, plain_big: pd.DataFrame) -> None:
    """Scaling-crossover table: DAG best vs KnapsackDP baseline per num_items.

    The crossover question: with the same capacity (1e6, table far above
    cache), does the DAG's wall-clock scaling improve as num_items -- and
    therefore the work per tile and between synchronizations -- grows?
    """
    dag = best_big[best_big["family"] == 10].set_index(["num_items", "processors"])
    plain = plain_big[plain_big["max_weight"] == 10]
    rows = []
    for (ni, proc), drow in dag.iterrows():
        prow = plain[(plain["num_items"] == ni) & (plain["processors"] == proc)]
        if prow.empty:
            continue
        prow = prow.iloc[0]
        rows.append({
            "num_items": ni,
            "processors": proc,
            "dag_cfg": f"ib{int(drow['item_block'])}_cb{int(drow['cap_block'])}",
            "t_dag": drow["time"],
            "s_wall_dag": drow["s_wall"],
            "t_plain": prow["time"],
            "s_wall_plain": prow["speedup"],
            "t_dag/t_plain": drow["time"] / prow["time"],
            "s_cyc_dag": drow["s_cycles"],
            "insn_ratio_dag": drow["insn_ratio"],
            "bw_dag_gbs": drow["bw_gbs"],
            "bw_plain_gbs": prow["bw_gbs"],
            "coverage_pct": drow["coverage_pct"],
        })
    table = pd.DataFrame(rows)
    print("\n=== Crossover sweep (capacity=1e6, max_weight=10): DAG best vs plain DP ===")
    print(table.round(4).to_string(index=False))


def plot_big_crossover(
    kpdp: pd.DataFrame,
    best_small: pd.DataFrame,
    plain_big: pd.DataFrame,
    best_big: pd.DataFrame,
    out: Path,
) -> None:
    """Wall speedup vs threads, one panel per num_items (50 from the n=50 sweep).

    The crossover experiment in one figure: if the DAG's scaling limit is
    the synchronization-to-work ratio, the DAG curve must lift toward (or
    past) the plain-DP curve as num_items -- work per tile and between
    synchronizations -- grows by 5x/20x/100x.
    """
    nis = [50, 1000, 5000]
    fig, axes = plt.subplots(1, len(nis), figsize=(4.4 * len(nis), 4.4), sharey=True)
    for ax, ni in zip(axes, nis):
        if ni == 50:
            dag = best_small[
                (best_small["family"] == 10) & (best_small["capacity"] == 1_000_000)
            ]
            plain = kpdp[
                (kpdp["family"] == 10) & (kpdp["capacity"] == 1_000_000)
            ]
        else:
            dag = best_big[best_big["num_items"] == ni]
            plain = plain_big[
                (plain_big["num_items"] == ni) & (plain_big["max_weight"] == 10)
            ]
        procs = sorted(set(dag["processors"]) | set(plain["processors"]))
        ax.plot(plain["processors"], plain["speedup"], "o-",
                color="#1f77b4", label="KnapsackDP")
        ax.plot(dag["processors"], dag["s_wall"], "s-",
                color="#d62728", label="KnapsackDPDAG (best)")
        _style_axis_threads(ax, procs)
        ax.set_xlabel("processors (threads)")
        ax.set_title(f"num_items={ni}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("wall speedup S_wall = T(1)/T(p)")
    fig.suptitle("Scaling crossover: DAG vs plain DP at capacity=1e6 (best config per p)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_bandwidth(
    kpdp: pd.DataFrame,
    best_small: pd.DataFrame,
    plain_big: pd.DataFrame,
    best_big: pd.DataFrame,
    out: Path,
) -> None:
    """Estimated DRAM bandwidth (GB/s) vs threads.

    Left: the n=50 campaign -- plain DP per (family, capacity) plus the DAG
    best configurations.  Right: the crossover sweep at capacity=1e6, one
    line per num_items for the plain baseline and the DAG best.  Flat
    curves at high p indicate DRAM bandwidth saturation; still-growing
    curves indicate the limit lies elsewhere (latency, synchronization).
    The estimate is a lower bound on reads plus a one-write-per-cell model
    (see add_bandwidth).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax = axes[0]
    for fam in sorted(kpdp["family"].unique()):
        for cap in sorted(kpdp["capacity"].unique()):
            curve = kpdp[(kpdp["family"] == fam) & (kpdp["capacity"] == cap)]
            curve = curve.sort_values("processors")
            ax.plot(curve["processors"], curve["bw_gbs"], "o",
                    color=FAMILY_COLORS[fam],
                    linestyle="-" if cap == 1_000_000 else "--",
                    label=f"DP w={fam}, cap={cap:,}")
    for fam in sorted(best_small["family"].unique()):
        curve = best_small[best_small["capacity"] == 1_000_000].sort_values("processors")
        ax.plot(curve["processors"], curve["bw_gbs"], "s--", alpha=0.75,
                color=FAMILY_COLORS[fam], label=f"DAG best w={fam}, cap=1e6")
    _style_axis_threads(ax, sorted(kpdp["processors"].unique()))
    ax.set_xlabel("processors (threads)")
    ax.set_ylabel("estimated DRAM bandwidth (GB/s)")
    ax.set_title("n=50 campaign")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[1]
    for ni in sorted(N_COLORS):
        curve = plain_big[(plain_big["num_items"] == ni)
                          & (plain_big["max_weight"] == 10)].sort_values("processors")
        ax.plot(curve["processors"], curve["bw_gbs"], "o-",
                color=N_COLORS[ni], label=f"DP n={ni}")
        curve = best_big[best_big["num_items"] == ni].sort_values("processors")
        ax.plot(curve["processors"], curve["bw_gbs"], "s--",
                color=N_COLORS[ni], label=f"DAG best n={ni}")
    _style_axis_threads(ax, sorted(plain_big["processors"].unique()))
    ax.set_xlabel("processors (threads)")
    ax.set_title("crossover sweep, capacity=1e6, max_weight=10")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)
    fig.suptitle("Estimated DRAM bandwidth vs threads (read misses x 64B + one 4B write per cell)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_kpdp_ipc(merged: pd.DataFrame, out: Path) -> None:
    """IPC(p) = instructions/cycles vs threads, one panel per weight family."""
    families = sorted(merged["family"].unique())
    fig, axes = plt.subplots(1, len(families), figsize=(5.5 * len(families), 4.6), sharey=True)
    if len(families) == 1:
        axes = [axes]
    for ax, family in zip(axes, families):
        sub = merged[merged["family"] == family]
        for cap in sorted(sub["capacity"].unique()):
            curve = sub[sub["capacity"] == cap].sort_values("processors")
            ax.plot(curve["processors"], curve["ipc"], "o-",
                    color=CAP_COLORS[cap], label=f"capacity={cap:,}")
        _style_axis_threads(ax, sorted(sub["processors"].unique()))
        ax.set_xlabel("processors (threads)")
        ax.set_title(f"max_weight={family}")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("IPC  instructions/cycle")
    fig.suptitle("KnapsackDP instructions per cycle vs threads")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_kpdp_memory_intensity(merged: pd.DataFrame, out: Path) -> None:
    """Miss intensity MPKI_e(p) = miss_e/instructions*1000 vs threads.

    Left: L1 misses; right: LLC (ll_cache) misses.  Colour = weight family,
    linestyle = capacity.
    """
    families = sorted(merged["family"].unique())
    caps = sorted(merged["capacity"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, col, label in ((axes[0], "mpki_l1", "L1-dcache misses"),
                           (axes[1], "mpki_llc", "LLC misses")):
        for family in families:
            for cap in caps:
                curve = merged[(merged["family"] == family) & (merged["capacity"] == cap)]
                curve = curve.sort_values("processors")
                ax.plot(curve["processors"], curve[col],
                        "-" if cap == caps[0] else "--",
                        color=FAMILY_COLORS[family],
                        label=f"max_weight={family}, cap={cap:,}")
        _style_axis_threads(ax, sorted(merged["processors"].unique()))
        ax.set_xlabel("processors (threads)")
        ax.set_title(label)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("MPKI  misses / 1000 instructions")
    fig.suptitle("KnapsackDP memory intensity vs threads")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_kpdp_cycles_per_cell(merged: pd.DataFrame, out: Path) -> None:
    """K_c(w, c, p) = cycles / ((n+1)(c+1)) vs capacity, one panel per family.

    A flat serial line means constant work per cell; growth with capacity or
    with the thread count exposes memory-bound behaviour.
    """
    families = sorted(merged["family"].unique())
    procs = sorted(merged["processors"].unique())
    fig, axes = plt.subplots(1, len(families), figsize=(5.5 * len(families), 4.6), sharey=True)
    if len(families) == 1:
        axes = [axes]
    for ax, family in zip(axes, families):
        sub = merged[merged["family"] == family]
        for proc in procs:
            curve = sub[sub["processors"] == proc].sort_values("capacity")
            ax.plot(curve["capacity"], curve["cycles_per_cell"], "o-",
                    color=THREAD_COLORS[proc], label=f"threads={proc}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("capacity (log)")
        ax.set_title(f"max_weight={family}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("cycles per DP cell (log)")
    fig.suptitle("KnapsackDP hardware work per DP table cell")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_kpdp_tlb_memory_wall(merged: pd.DataFrame, out: Path) -> None:
    """dTLB pressure vs cycles per DP cell, one column per weight family.

    Top row: dTLB MPKI vs capacity.  Bottom row: cycles per DP cell vs
    capacity (log-log).  The dTLB miss rate stays ~3-5 MPKI across the whole
    capacity range for every thread count, while K_c grows ~4x with the
    thread count: the per-cell degradation is contention-related (spin,
    cache-line sharing, PMU-visible overhead cycles summed over cores), NOT
    TLB-driven, even though the cap=1e7 table (~400 MB) far exceeds the TLB
    reach.
    """
    families = sorted(merged["family"].unique())
    procs = sorted(merged["processors"].unique())
    fig, axes = plt.subplots(2, len(families), figsize=(5.5 * len(families), 8),
                             sharey="row", squeeze=False)
    for j, family in enumerate(families):
        sub = merged[merged["family"] == family]
        for proc in procs:
            curve = sub[sub["processors"] == proc].sort_values("capacity")
            axes[0][j].plot(curve["capacity"], curve["mpki_dtlb"], "o-",
                            color=THREAD_COLORS[proc], label=f"threads={proc}")
            axes[1][j].plot(curve["capacity"], curve["cycles_per_cell"], "o-",
                            color=THREAD_COLORS[proc])
        for i in range(2):
            axes[i][j].set_xscale("log")
            axes[i][j].set_yscale("log")
            axes[i][j].set_xlabel("capacity (log)")
            axes[i][j].grid(True, which="both", alpha=0.3)
        axes[0][j].set_title(f"max_weight={family}")
        axes[0][j].legend(fontsize=8)
    axes[0][0].set_ylabel("dTLB MPKI (log)")
    axes[1][0].set_ylabel("cycles per DP cell (log)")
    fig.suptitle("KnapsackDP: dTLB pressure (flat) vs cycles per cell (grows with threads)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_decomposition(merged: pd.DataFrame, wall_col: str, title: str, out: Path) -> None:
    """Speedup decomposition at max threads: S_wall = S_cyc * p * r.

    For every weight family and capacity three bars: wall speedup (same-run
    median times), cycle-count reduction S_cyc (perf, summed over all cores)
    and the
    matching multiplier p * r with r = per-core cycle-rate ratio, so that
    bar 1 = bar 2 * bar 3 by construction.
    """
    families = sorted(merged["family"].unique())
    top = merged[merged["processors"] == merged["processors"].max()].copy()
    top["p_rate"] = top["processors"] * top["rate_ratio"]
    width = 0.25
    fig, axes = plt.subplots(1, len(families), figsize=(5.5 * len(families), 4.6), sharey=True)
    if len(families) == 1:
        axes = [axes]
    for ax, family in zip(axes, families):
        sub = top[top["family"] == family].sort_values("capacity")
        caps = sorted(sub["capacity"].unique())
        for i, (col, label) in enumerate(
            ((wall_col, "S_wall"), ("s_cycles", "S_cyc"), ("p_rate", "p·r"))
        ):
            ax.bar(np.arange(len(caps)) + (i - 1) * width, sub[col], width,
                   color=("#1f77b4", "#ff7f0e", "#2ca02c")[i], label=label)
        ax.set_xticks(np.arange(len(caps)))
        ax.set_xticklabels([f"{c:,}" for c in caps])
        ax.set_title(f"max_weight={family}")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel(f"factor at {int(merged['processors'].max())} threads")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_dag_block_grid(configs: pd.DataFrame, out: Path) -> None:
    """(item_block x cap_block) heatmaps at BLOCK_GRID_POINT.

    Left: median wall time (log colour).  Middle: IPC.  Right: LLC MPKI.
    Shows how tile size trades scheduling overhead against cache locality.
    """
    point = BLOCK_GRID_POINT
    sub = configs[
        (configs["family"] == point["family"])
        & (configs["capacity"] == point["capacity"])
        & (configs["processors"] == point["processors"])
    ]
    cbs = sorted(sub["cap_block"].unique())
    ibs = sorted(sub["item_block"].unique(), reverse=True)
    panels = [("time", "wall time (s)", True), ("ipc", "IPC", False), ("mpki_llc", "LLC MPKI", False)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (col, label, log) in zip(axes, panels):
        grid = sub.pivot(index="item_block", columns="cap_block", values=col).reindex(
            index=ibs, columns=cbs
        )
        norm = matplotlib.colors.LogNorm(vmin=grid.min().min(), vmax=grid.max().max()) if log else None
        image = ax.imshow(grid.values, aspect="auto", cmap="viridis", norm=norm)
        ax.set_xticks(range(len(cbs)))
        ax.set_xticklabels([str(c) for c in cbs])
        ax.set_yticks(range(len(ibs)))
        ax.set_yticklabels([str(i) for i in ibs])
        for row in range(grid.shape[0]):
            for col_i in range(grid.shape[1]):
                value = grid.values[row, col_i]
                ax.text(col_i, row, f"{value:.4g}" if np.isfinite(value) else "-",
                        ha="center", va="center", fontsize=9,
                        color="white")
        ax.set_xlabel("cap_block")
        ax.set_title(label)
        fig.colorbar(image, ax=ax, fraction=0.046)
    axes[0].set_ylabel("item_block")
    fig.suptitle(
        f"KnapsackDPDAG tile-size effect (max_weight={point['family']}, "
        f"capacity={point['capacity']:,}, threads={point['processors']})"
    )
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_dag_miss_vs_tiles(configs: pd.DataFrame, out: Path) -> None:
    """Cache behaviour vs tile count at p=1 (max_weight=10).

    Top row: LLC MPKI vs tiles; bottom row: median wall time vs tiles.
    Columns: capacity 1e5 and 1e6.  Colour = cap_block, marker = item_block.
    Small cap_blocks fragment the table into many tiny tiles and destroy
    locality; large ones approach the untiled behaviour.
    """
    serial = configs[(configs["family"] == 10) & (configs["processors"] == 1)]
    caps = [100_000, 1_000_000]
    fig, axes = plt.subplots(2, len(caps), figsize=(12, 8), sharex="col")
    for j, cap in enumerate(caps):
        sub = serial[serial["capacity"] == cap]
        for cb in sorted(sub["cap_block"].unique()):
            for ib in sorted(sub["item_block"].unique()):
                row = sub[(sub["cap_block"] == cb) & (sub["item_block"] == ib)]
                if row.empty:
                    continue
                axes[0][j].plot(row["tiles"], row["mpki_llc"], IB_MARKERS[ib],
                                color=BLOCK_COLORS[cb], label=f"cb={cb}, ib={ib}")
                axes[1][j].plot(row["tiles"], row["time"], IB_MARKERS[ib],
                                color=BLOCK_COLORS[cb], label=f"cb={cb}, ib={ib}")
        for i in range(2):
            axes[i][j].set_xscale("log")
            axes[i][j].set_yscale("log")
            axes[i][j].set_xlabel("tiles (log)")
            axes[i][j].set_title(f"capacity={cap:,}")
            axes[i][j].grid(True, which="both", alpha=0.3)
        axes[0][j].legend(fontsize=7)
    axes[0][0].set_ylabel("LLC MPKI (log)")
    axes[1][0].set_ylabel("wall time (s, log)")
    fig.suptitle("KnapsackDPDAG: tile count vs cache misses and time (max_weight=10, p=1)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_dag_best_config_map(best: pd.DataFrame, out: Path) -> None:
    """Which (item_block/cap_block) wins per (family, capacity, threads).

    One panel per weight family; cells coloured by the winning tile
    configuration and annotated with item_block/cap_block.  NOTE: at these
    run lengths (<100 ms) the ranking by median wall time is partly
    noise-dominated (see print_dag_reliability), so read this as a summary
    of the time ranking, not a stable optimum.
    """
    code = {
        (5, 1000): 0, (5, 10000): 1, (10, 1000): 2, (10, 10000): 3,
        (15, 1000): 4, (15, 10000): 5, (25, 1000): 6, (25, 10000): 7,
    }
    fams = sorted(best["family"].unique())
    caps = sorted(best["capacity"].unique())
    procs = sorted(best["processors"].unique())
    cmap = matplotlib.colors.ListedColormap(plt.cm.viridis(np.linspace(0, 1, len(code))))
    fig, axes = plt.subplots(1, len(fams), figsize=(6.5 * len(fams), 3.8), squeeze=False)
    for ax, fam in zip(axes[0], fams):
        grid = np.full((len(caps), len(procs)), np.nan)
        text = [[""] * len(procs) for _ in caps]
        for _, row in best[best["family"] == fam].iterrows():
            i = caps.index(row["capacity"])
            j = procs.index(row["processors"])
            grid[i, j] = code[(int(row["item_block"]), int(row["cap_block"]))]
            text[i][j] = f"ib{int(row['item_block'])}\ncb{int(row['cap_block'])}"
        ax.imshow(grid, cmap=cmap, vmin=-0.5, vmax=len(code) - 0.5, aspect="auto")
        ax.set_xticks(range(len(procs)))
        ax.set_xticklabels([str(p) for p in procs])
        ax.set_yticks(range(len(caps)))
        ax.set_yticklabels([f"{c:,}" for c in caps])
        for i in range(len(caps)):
            for j in range(len(procs)):
                if text[i][j]:
                    ax.text(j, i, text[i][j], ha="center", va="center",
                            fontsize=8, color="white")
        ax.set_xlabel("processors (threads)")
        ax.set_title(f"max_weight={fam}")
    axes[0][0].set_ylabel("capacity")
    fig.suptitle("KnapsackDPDAG fastest tile configuration (item_block / cap_block)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_dag_reliability(configs: pd.DataFrame, best: pd.DataFrame, out: Path) -> None:
    """Left: run duration vs PMU coverage — short runs multiplex heavily.
    Right: best-config wall time vs threads — the flat lines show that the
    sweep's runs are too short for wall time to resolve parallel scaling;
    counter ratios (IPC, MPKI) are the robust signal there."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    for cap in sorted(configs["capacity"].unique()):
        sub = configs[configs["capacity"] == cap]
        ax.scatter(sub["time"], sub["coverage_pct"], s=14, alpha=0.7,
                   color=CAP_COLORS[cap], label=f"capacity={cap:,}")
    ax.axvline(0.1, color="gray", linestyle=":", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("wall time (s, log)")
    ax.set_ylabel("PMU coverage (%)")
    ax.set_title("run duration vs multiplexing coverage")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    for fam in sorted(best["family"].unique()):
        for cap in sorted(best["capacity"].unique()):
            curve = best[(best["family"] == fam) & (best["capacity"] == cap)]
            curve = curve.sort_values("processors")
            ax.plot(curve["processors"], curve["time"] * 1e3, "o",
                    color=CAP_COLORS[cap], linestyle="-" if fam == 10 else "--",
                    label=f"w={fam}, cap={cap:,}")
    _style_axis_threads(ax, sorted(best["processors"].unique()))
    ax.set_yscale("log")
    ax.set_xlabel("processors (threads)")
    ax.set_ylabel("best-config wall time (ms, log)")
    ax.set_title("best-config time vs threads (flat = noise-dominated)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.suptitle("KnapsackDPDAG measurement reliability")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_dag_ipc(best: pd.DataFrame, out: Path) -> None:
    """IPC(p) of the per-p best configuration, one panel per weight family."""
    families = sorted(best["family"].unique())
    fig, axes = plt.subplots(1, len(families), figsize=(5.5 * len(families), 4.6), sharey=True)
    if len(families) == 1:
        axes = [axes]
    for ax, family in zip(axes, families):
        sub = best[best["family"] == family]
        for cap in sorted(sub["capacity"].unique()):
            curve = sub[sub["capacity"] == cap].sort_values("processors")
            ax.plot(curve["processors"], curve["ipc"], "o-",
                    color=CAP_COLORS[cap], label=f"capacity={cap:,}")
        _style_axis_threads(ax, sorted(sub["processors"].unique()))
        ax.set_xlabel("processors (threads)")
        ax.set_title(f"max_weight={family}")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("IPC  instructions/cycle")
    fig.suptitle("KnapsackDPDAG instructions per cycle vs threads (best config)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_dag_speedup(best: pd.DataFrame, out: Path) -> None:
    """Wall speedup S_wall = T(1)/T(p) of the per-p best DAG configuration.

    Same layout as the classic KnapsackDP speedup figure (one panel per
    weight family, one line per capacity, ideal S = p reference) so the two
    chapters can be read side by side; only the DAG's own campaign is shown.
    """
    families = sorted(best["family"].unique())
    fig, axes = plt.subplots(1, len(families), figsize=(5.5 * len(families), 4.6), sharey=True)
    if len(families) == 1:
        axes = [axes]
    for ax, family in zip(axes, families):
        sub = best[best["family"] == family]
        procs = sorted(sub["processors"].unique())
        for cap in sorted(sub["capacity"].unique()):
            curve = sub[sub["capacity"] == cap].sort_values("processors")
            ax.plot(curve["processors"], curve["s_wall"], "o-",
                    color=CAP_COLORS[cap], label=f"capacity={cap:,}")
        ax.plot(procs, procs, "k--", linewidth=1, label="ideal")
        _style_axis_threads(ax, procs)
        ax.set_xlabel("processors (threads)")
        ax.set_title(f"max_weight={family}")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("wall speedup S_wall = T(1)/T(p)")
    fig.suptitle("KnapsackDPDAG speedup vs thread count (best config)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_comparison(kpdp: pd.DataFrame, best: pd.DataFrame, out: Path) -> None:
    """Bars for KnapsackDP vs KnapsackDPDAG-best on the common grid.

    Rows: weight family; columns: IPC, cycles per DP cell, LLC MPKI and the
    TOTAL LLC misses per run (log); x: the common thread counts; two bars
    each (plain DP vs best DAG configuration at capacity=1e6).  The total
    panel makes the tiling effect explicit: ~20x fewer LLC misses at equal
    instruction counts.
    """
    metrics = [
        ("ipc", "IPC"),
        ("cycles_per_cell", "cycles per DP cell"),
        ("mpki_llc", "LLC MPKI"),
        ("ll_cache_miss_rd", "LLC misses per run (log)"),
    ]
    width = 0.35
    fig, axes = plt.subplots(len(CMP_FAMILIES), len(metrics),
                             figsize=(18, 4.2 * len(CMP_FAMILIES)), squeeze=False)
    for i, family in enumerate(CMP_FAMILIES):
        k = kpdp[(kpdp["family"] == family) & (kpdp["capacity"] == CMP_CAP)]
        d = best[(best["family"] == family) & (best["capacity"] == CMP_CAP)]
        for j, (col, label) in enumerate(metrics):
            ax = axes[i][j]
            for t_i, proc in enumerate(CMP_THREADS):
                kv = k[k["processors"] == proc][col]
                dv = d[d["processors"] == proc][col]
                ax.bar(t_i - width / 2, kv.iloc[0] if len(kv) else np.nan, width,
                       color="#1f77b4", label="KnapsackDP" if t_i == 0 else None)
                ax.bar(t_i + width / 2, dv.iloc[0] if len(dv) else np.nan, width,
                       color="#d62728", label="KnapsackDPDAG (best)" if t_i == 0 else None)
            if col == "ll_cache_miss_rd":
                ax.set_yscale("log")
            ax.set_xticks(range(len(CMP_THREADS)))
            ax.set_xticklabels([str(t) for t in CMP_THREADS])
            ax.set_xlabel("threads")
            ax.set_ylabel(label)
            ax.set_title(f"max_weight={family}")
            ax.grid(True, axis="y", alpha=0.3)
            ax.legend(fontsize=8)
    fig.suptitle(f"Plain DP vs tiled DAG DP at capacity={CMP_CAP:,}")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    perf_kpdp, long_kpdp = load_perf_csv(KPDP_PERF_FILE)
    perf_dag, long_dag = load_perf_csv(DAG_PERF_FILE)
    perf_big, long_big = load_perf_csv(BIG_PERF_FILE)
    crosscheck_metrics(long_kpdp, "perf_kpdp.csv")
    crosscheck_metrics(long_dag, "perf_kpdpdag.csv")
    crosscheck_metrics(long_big, "perf_kpdpbig.csv")
    report_coverage(perf_kpdp, "perf_kpdp.csv")
    report_coverage(perf_dag, "perf_kpdpdag.csv")
    report_coverage(perf_big, "perf_kpdpbig.csv")
    print_repeatability(perf_kpdp, "perf_kpdp.csv")
    print_repeatability(perf_dag, "perf_kpdpdag.csv")
    print_repeatability(perf_big, "perf_kpdpbig.csv")

    kpdp = build_kpdp(perf_kpdp)
    configs, best = build_dag(perf_dag)
    configs_big, best_big = build_dag(perf_big[perf_big["test"] == "knapsackdpdag"])
    plain_big = build_big_plain(perf_big)

    print_kpdp_tables(kpdp)
    print_dag_tables(configs, best)
    print_dag_reliability(configs, best)
    print_comparison_table(kpdp, best)
    print_big_tables(best_big, plain_big)
    print_clock_check(kpdp, "KnapsackDP cycles-vs-time check")
    print_clock_check(configs, "KnapsackDPDAG cycles-vs-time check")
    print_clock_check(configs_big, "KnapsackDPDAG (crossover sweep) cycles-vs-time check")
    print_clock_check(plain_big, "KnapsackDP (crossover sweep) cycles-vs-time check")

    plots = {
        "perf_kpdp_ipc_vs_threads.png": lambda: plot_kpdp_ipc(kpdp, OUT_DIR / "perf_kpdp_ipc_vs_threads.png"),
        "perf_kpdp_memory_intensity.png": lambda: plot_kpdp_memory_intensity(kpdp, OUT_DIR / "perf_kpdp_memory_intensity.png"),
        "perf_kpdp_cycles_per_cell.png": lambda: plot_kpdp_cycles_per_cell(kpdp, OUT_DIR / "perf_kpdp_cycles_per_cell.png"),
        "perf_kpdp_tlb_memory_wall.png": lambda: plot_kpdp_tlb_memory_wall(kpdp, OUT_DIR / "perf_kpdp_tlb_memory_wall.png"),
        "perf_kpdp_decomposition.png": lambda: plot_decomposition(kpdp, "speedup", "KnapsackDP speedup decomposition (S_wall = S_cyc * p * r)", OUT_DIR / "perf_kpdp_decomposition.png"),
        "perf_kpdpdag_block_grid.png": lambda: plot_dag_block_grid(configs, OUT_DIR / "perf_kpdpdag_block_grid.png"),
        "perf_kpdpdag_miss_vs_tiles.png": lambda: plot_dag_miss_vs_tiles(configs, OUT_DIR / "perf_kpdpdag_miss_vs_tiles.png"),
        "perf_kpdpdag_best_config_map.png": lambda: plot_dag_best_config_map(best, OUT_DIR / "perf_kpdpdag_best_config_map.png"),
        "perf_kpdpdag_reliability.png": lambda: plot_dag_reliability(configs, best, OUT_DIR / "perf_kpdpdag_reliability.png"),
        "perf_kpdpdag_ipc_vs_threads.png": lambda: plot_dag_ipc(best, OUT_DIR / "perf_kpdpdag_ipc_vs_threads.png"),
        "perf_kpdpdag_speedup.png": lambda: plot_dag_speedup(best, OUT_DIR / "perf_kpdpdag_speedup.png"),
        "perf_kpdpdag_decomposition.png": lambda: plot_decomposition(best, "s_wall", "KnapsackDPDAG speedup decomposition (best config, S_wall = S_cyc * p * r)", OUT_DIR / "perf_kpdpdag_decomposition.png"),
        "perf_vs_algorithms.png": lambda: plot_comparison(kpdp, best, OUT_DIR / "perf_vs_algorithms.png"),
        "perf_big_crossover.png": lambda: plot_big_crossover(kpdp, best, plain_big, best_big, OUT_DIR / "perf_big_crossover.png"),
        "perf_bandwidth.png": lambda: plot_bandwidth(kpdp, best, plain_big, best_big, OUT_DIR / "perf_bandwidth.png"),
    }
    for name, func in plots.items():
        func()
        print(f"saved {OUT_DIR / name}")


if __name__ == "__main__":
    main()
