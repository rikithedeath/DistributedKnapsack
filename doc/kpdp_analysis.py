"""Analysis of doc/data/result_kpdp.json (parallel KnapsackDP benchmark).

The file contains sequential/shared-memory KnapsackDP runs with 50 items,
seed=42, host epito02, for three problem families distinguished by max_weight
(10 / 100 / 1000).  Each family is executed at 6 thread counts
(processors = 1, 2, 4, 8, 16, 32); inside every (family, processors) block the
capacities [1e4, 1e5, 1e6, 1e7, 1e7] are measured, the 1e7 run being duplicated.

Because the processor count is now populated from OMP_NUM_THREADS this script
computes the classic parallel metrics, plus the effect of capacity and of the
weight distribution (max_weight), the short-term repeatability of the duplicated
1e7 runs and the per-item section breakdown from time_sections.

Notation used throughout (all formulas are also repeated in each plot):

    p                     number of processors/threads
    w                     weight-range family, w = max_weight in {10, 100, 1000}
    c                     knapsack capacity
    T(p)                  wall-clock time (s) at p threads, averaged over repeats
    T_w(c, p)             wall time restricted to weight family w and capacity c
    K(p)                  "KnapsackDP::Compute" section time (ms), averaged
    S(p) = T(1)/T(p)                            parallel speedup
    E(p) = S(p)/p                               parallel efficiency
    C(p) = p * T(p)                             parallel cost
    S_K(p) = K(1)/K(p)                          kernel (Compute-section) speedup
    E_K(p) = S_K(p)/p                           kernel efficiency
    R_s(p) = K_s(p)/K_s(1)                      relative time of section s
                                                (1.0 = serial value)
    phi(p) = K(p) / (T(p)*1000)                 Compute share of wall time
    overhead(p) = T(p)*1000 - K(p)              non-compute wall time (ms)
    I(w, c, p) = max_item(w,c,p) / mean_item(w,c,p)   per-item load imbalance
    delta(w, c, p) = R(w,c,p) / R(w=10,c,p)     weight-range relative cost

Serial capacity scaling is fitted on the log-log line
    log T(1) = a * log c + b          (slope a ~ 1 expected for O(N*c) DP).

Plots are written to doc/images/ with a "kpdp_" prefix.

Run with:  ~/pyenv/sci/bin/python doc/kpdp_analysis.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors
import matplotlib.lines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "result_kpdp.json"
OUT_DIR = BASE_DIR / "images"

FAMILY_ORDER = [10, 100, 1000]
FAMILY_COLORS = {10: "#1f77b4", 100: "#ff7f0e", 1000: "#2ca02c"}
CAP_COLORS = {
    10_000: "#1f77b4",
    100_000: "#ff7f0e",
    1_000_000: "#2ca02c",
    10_000_000: "#d62728",
}

SECTIONS = (
    "KnapsackDP::Compute",
    "KnapsackDP::ComputeItem",
    "KnapsackDP::ComputeThreadBlock",
)


def load_data() -> pd.DataFrame:
    with DATA_FILE.open() as fh:
        records = json.load(fh)

    df = pd.DataFrame(records)
    sections = df.pop("time_sections")

    for section in SECTIONS:
        col = section.replace("::", "_")
        for stat in ("count", "mean", "min", "max", "variance"):
            df[f"{col}_{stat}"] = sections.apply(lambda s, c=section, st=stat: s[c][stat])
        # Total time spent in the section (mean * number of samples); all in ms.
        df[f"{col}_total"] = df[f"{col}_mean"] * df[f"{col}_count"]

    df["family"] = df["max_weight"]
    df["capacity"] = df["capacity"].astype(int)
    df["processors"] = df["processors"].astype(int)
    df["time_ms"] = df["time"] * 1000.0

    # Index of the duplicated run inside each (family, processors) block.
    run_in_block = []
    seen = 0
    prev_key = None
    for _, row in df.iterrows():
        key = (row["family"], row["processors"])
        if key != prev_key:
            seen = 0
            prev_key = key
        seen += 1
        run_in_block.append(seen)
    df["run_in_block"] = run_in_block
    return df.sort_values(["family", "processors", "capacity"]).reset_index(drop=True)


def add_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Attach per-(family,capacity) baselines, speedups, efficiency and section data.

    Formulas computed here (per family and capacity):

        T(p)      = mean over repeats of the wall-clock time
        K(p)      = mean of KnapsackDP::Compute (ms); serial baseline K(1)
        S(p)      = T(1) / T(p)                     wall speedup
        E(p)      = S(p) / p                        wall efficiency
        C(p)      = p * T(p)                        parallel cost
        S_K(p)    = K(1) / K(p)                     Compute-section speedup
        E_K(p)    = S_K(p) / p                      Compute-section efficiency
        time_ms   = T(p) * 1000
        overhead  = time_ms - K(p)                  (ms of non-compute wall time)
        phi(p)    = K(p) / (T(p) * 1000)            Compute fraction of wall time
        R_s(p)    = K_s(p) / K_s(1)                 relative section time,
                    for s in {Compute, ComputeItem, ComputeThreadBlock}
    """
    # Mean time per (family, capacity, processors), averaging duplicate runs.
    grouped = (
        df.groupby(["family", "capacity", "processors"])
        .agg(
            time=("time", "mean"),
            time_std=("time", "std"),
            n=("time", "count"),
            compute_ms=("KnapsackDP_Compute_total", "mean"),
            item_ms=("KnapsackDP_ComputeItem_total", "mean"),
            tb_ms=("KnapsackDP_ComputeThreadBlock_total", "mean"),
        )
        .reset_index()
    )
    base = grouped[grouped["processors"] == 1].set_index(["family", "capacity"])
    grouped = grouped.merge(
        base["time"].rename("t_base"), on=["family", "capacity"]
    ).merge(
        base["compute_ms"].rename("compute_base"), on=["family", "capacity"]
    )
    grouped["speedup"] = grouped["t_base"] / grouped["time"]
    grouped["efficiency"] = grouped["speedup"] / grouped["processors"]
    grouped["cost"] = grouped["processors"] * grouped["time"]

    grouped["compute_speedup"] = grouped["compute_base"] / grouped["compute_ms"]
    grouped["compute_efficiency"] = grouped["compute_speedup"] / grouped["processors"]
    grouped["time_ms"] = grouped["time"] * 1000.0
    grouped["overhead_ms"] = grouped["time_ms"] - grouped["compute_ms"]
    grouped["compute_fraction"] = grouped["compute_ms"] / grouped["time_ms"]

    # Relative time of every section w.r.t. its serial value (1.0 == serial).
    for col, name in (
        ("compute_ms", "compute_rel"),
        ("item_ms", "item_rel"),
        ("tb_ms", "tb_rel"),
    ):
        base_col = base[col].rename(f"{col}_base")
        grouped = grouped.merge(base_col, on=["family", "capacity"])
        grouped[name] = grouped[col] / grouped[f"{col}_base"]
        grouped = grouped.drop(columns=[f"{col}_base"])
    return grouped


def print_summary(metrics: pd.DataFrame) -> None:
    pd.set_option("display.width", 180)
    pd.set_option("display.max_columns", 25)

    print("\n=== Time / speedup / efficiency per (family, capacity, processors) ===")
    table = metrics.sort_values(["family", "capacity", "processors"])[
        ["family", "capacity", "processors", "time", "speedup", "efficiency", "cost",
         "compute_ms", "compute_speedup", "compute_efficiency", "compute_fraction"]
    ]
    print(table.round(6).to_string(index=False))

    print("\n=== Wall vs Compute speedup at max processors ===")
    top_p = metrics["processors"].max()
    top = metrics[metrics["processors"] == top_p]
    wall = top.pivot(index="capacity", columns="family", values="speedup")
    comp = top.pivot(index="capacity", columns="family", values="compute_speedup")
    cf = top.pivot(index="capacity", columns="family", values="compute_fraction")
    print(pd.concat({"wall_speedup": wall, "compute_speedup": comp,
                     "compute_fraction": cf}, axis=1).round(4).to_string())

    print("\n=== Best speedup (largest processors) vs capacity ===")
    top_p = metrics["processors"].max()
    top = metrics[metrics["processors"] == top_p]
    pivot = top.pivot(index="capacity", columns="family", values="speedup")
    eff = top.pivot(index="capacity", columns="family", values="efficiency")
    summary = pd.concat({"speedup": pivot, "efficiency": eff}, axis=1)
    print(summary.round(4).to_string())

    print("\n=== Speedup at every processor count (family=max_weight=10) ===")
    ref = metrics[metrics["family"] == 10].pivot(
        index="capacity", columns="processors", values="speedup"
    )
    print(ref.round(4).to_string())


def plot_speedup_vs_processors(metrics: pd.DataFrame, out: Path) -> None:
    """Speedup S(p) = T(1)/T(p) vs processors, one line per capacity.

    Ideal reference line is S_ideal(p) = p.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    procs = sorted(metrics["processors"].unique())
    for ax, family in zip(axes, FAMILY_ORDER):
        sub = metrics[metrics["family"] == family]
        for cap in sorted(sub["capacity"].unique()):
            curve = sub[sub["capacity"] == cap].sort_values("processors")
            ax.plot(
                curve["processors"],
                curve["speedup"],
                marker="o",
                color=CAP_COLORS[cap],
                label=f"capacity={cap:,}",
            )
        ax.plot(procs, procs, "k--", linewidth=1, label="ideal")
        ax.set_xscale("log", base=2)
        ax.set_xticks(procs)
        ax.set_xticklabels([str(p) for p in procs])
        ax.set_xlabel("processors (threads)")
        ax.set_title(f"max_weight={family}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("speedup  T(1)/T(p)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_efficiency_vs_processors(metrics: pd.DataFrame, out: Path) -> None:
    """Efficiency E(p) = S(p)/p = T(1)/(p*T(p)) vs processors.

    Ideal reference line is E_ideal(p) = 1.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    procs = sorted(metrics["processors"].unique())
    for ax, family in zip(axes, FAMILY_ORDER):
        sub = metrics[metrics["family"] == family]
        for cap in sorted(sub["capacity"].unique()):
            curve = sub[sub["capacity"] == cap].sort_values("processors")
            ax.plot(
                curve["processors"],
                curve["efficiency"],
                marker="o",
                color=CAP_COLORS[cap],
                label=f"capacity={cap:,}",
            )
        ax.axhline(1.0, color="k", linewidth=1, linestyle="--", label="ideal")
        ax.set_xscale("log", base=2)
        ax.set_xticks(procs)
        ax.set_xticklabels([str(p) for p in procs])
        ax.set_xlabel("processors (threads)")
        ax.set_ylim(0, 1.1)
        ax.set_title(f"max_weight={family}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("efficiency  S(p)/p")
    fig.suptitle("KnapsackDP parallel efficiency")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_speedup_heatmap(metrics: pd.DataFrame, out: Path) -> None:
    """Heatmap of S(c, p) = T_w(c, 1) / T_w(c, p) over capacity x processors.

    Each cell shows the speedup of weight family w at capacity c with p threads.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    procs = sorted(metrics["processors"].unique())
    caps = sorted(metrics["capacity"].unique())
    for ax, family in zip(axes, FAMILY_ORDER):
        sub = metrics[metrics["family"] == family]
        pivot = sub.pivot(index="capacity", columns="processors", values="speedup").reindex(
            index=caps, columns=procs
        )
        image = ax.imshow(pivot.values, aspect="auto", cmap="viridis", vmin=1.0)
        ax.set_xticks(range(len(procs)))
        ax.set_xticklabels([str(p) for p in procs])
        ax.set_yticks(range(len(caps)))
        ax.set_yticklabels([f"{c:,}" for c in caps])
        for row in range(pivot.shape[0]):
            for col in range(pivot.shape[1]):
                value = pivot.values[row, col]
                ax.text(
                    col, row, f"{value:.1f}", ha="center", va="center",
                    color="white" if value < pivot.values.max() * 0.6 else "black",
                    fontsize=9,
                )
        ax.set_xlabel("processors")
        ax.set_title(f"max_weight={family}")
    axes[0].set_ylabel("capacity")
    fig.colorbar(axes[-1].images[0], ax=axes, label="speedup", fraction=0.02)
    fig.suptitle("Speedup map: capacity x processors")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_time_vs_processors(metrics: pd.DataFrame, out: Path) -> None:
    """Raw wall time T_w(c, p) vs processors, one line per capacity.

    The ideal scaling is T_ideal(p) = T(1)/p.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    procs = sorted(metrics["processors"].unique())
    for ax, family in zip(axes, FAMILY_ORDER):
        sub = metrics[metrics["family"] == family]
        for cap in sorted(sub["capacity"].unique()):
            curve = sub[sub["capacity"] == cap].sort_values("processors")
            ax.plot(
                curve["processors"],
                curve["time"],
                marker="o",
                color=CAP_COLORS[cap],
                label=f"capacity={cap:,}",
            )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(procs)
        ax.set_xticklabels([str(p) for p in procs])
        ax.set_xlabel("processors (threads)")
        ax.set_title(f"max_weight={family}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("execution time (s, log)")
    fig.suptitle("KnapsackDP execution time vs thread count")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_capacity_effect(metrics: pd.DataFrame, out: Path) -> None:
    """Serial cost vs capacity and its log-log growth exponent.

    Left:  T_w(c, 1) vs c for each weight family.
    Right: least-squares fit  log T(1) = a * log c + b,  with mean T(1) over
           families; a ~ 1 matches the O(N*c) dynamic-programming complexity.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    procs = sorted(metrics["processors"].unique())

    for family in FAMILY_ORDER:
        sub = metrics[(metrics["family"] == family) & (metrics["processors"] == 1)].sort_values("capacity")
        axes[0].plot(
            sub["capacity"], sub["time"], "o-", color=FAMILY_COLORS[family], label=f"max_weight={family}"
        )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("capacity (log)")
    axes[0].set_ylabel("serial time T(1) (s, log)")
    axes[0].set_title("Serial cost grows ~linearly with capacity")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()

    caps = np.array(sorted(metrics["capacity"].unique()))
    serial = (
        metrics[metrics["processors"] == 1].groupby("capacity")["time"].mean().reindex(caps)
    )
    fit = np.polyfit(np.log(caps), np.log(serial.values), 1)
    axes[1].loglog(caps, serial.values, "o-", color="black", label="T(1) mean")
    axes[1].loglog(caps, np.exp(fit[1]) * caps**fit[0], "--", color="red",
                   label=f"fit slope = {fit[0]:.3f}")
    axes[1].set_xlabel("capacity (log)")
    axes[1].set_ylabel("serial time T(1) (s, log)")
    axes[1].set_title("Scaling exponent (O(N*C) -> slope ~ 1)")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_weight_effect(metrics: pd.DataFrame, out: Path) -> None:
    """Effect of the weight range (max_weight w) on execution time.

    Left:  T_w(c, 1) vs capacity for w in {10, 100, 1000}.
    Right: relative cost  T_w(c, p_max) / T_{10}(c, p_max)  at the largest
           thread count; 1.0 means identical to the narrowest range.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    caps = sorted(metrics["capacity"].unique())

    for family in FAMILY_ORDER:
        sub = metrics[metrics["family"] == family].sort_values(["capacity", "processors"])
        ref = sub[sub["processors"] == 1].set_index("capacity")["time"].reindex(caps)
        axes[0].plot(caps, ref.values, "o-", color=FAMILY_COLORS[family], label=f"max_weight={family}")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("capacity (log)")
    axes[0].set_ylabel("serial time T(1) (s, log)")
    axes[0].set_title("Weight distribution effect on serial performance")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()

    top_p = metrics["processors"].max()
    top = metrics[metrics["processors"] == top_p]
    base = top[top["family"] == 10].set_index("capacity")["time"].reindex(caps)
    for family in (100, 1000):
        curve = top[top["family"] == family].set_index("capacity")["time"].reindex(caps)
        axes[1].plot(caps, (curve / base).values, "o-", color=FAMILY_COLORS[family],
                     label=f"max_weight={family} / 10")
    axes[1].axhline(1.0, color="black", linewidth=0.8)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("capacity (log)")
    axes[1].set_ylabel("relative time at max processors")
    axes[1].set_title(f"Slowdown vs max_weight=10 at {top_p} processors")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_repeatability(df: pd.DataFrame, out: Path) -> None:
    """Short-term repeatability of the duplicated 1e7 runs.

    For each batch with two runs t1, t2:
        delta_rel = |t1 - t2| / ((t1 + t2)/2) * 100 %.
    """
    dup = df[df["capacity"] == 10_000_000]
    fig, ax = plt.subplots(figsize=(10, 5))
    procs = sorted(dup["processors"].unique())
    width = 0.25
    positions = np.arange(len(procs))
    for i, family in enumerate(FAMILY_ORDER):
        sub = dup[dup["family"] == family].sort_values("processors")
        rel = []
        for proc in procs:
            times = sub[sub["processors"] == proc]["time"].values
            if len(times) == 2:
                rel.append(abs(times[0] - times[1]) / times.mean() * 100.0)
            else:
                rel.append(0.0)
        ax.bar(positions + (i - 1) * width, rel, width, color=FAMILY_COLORS[family],
               label=f"max_weight={family}")
    ax.set_xticks(positions)
    ax.set_xticklabels([str(p) for p in procs])
    ax.set_xlabel("processors")
    ax.set_ylabel("relative difference between duplicated 1e7 runs (%)")
    ax.set_title("Short-term repeatability (capacity=1e7 measured twice)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_section_breakdown(metrics: pd.DataFrame, df: pd.DataFrame, out: Path) -> None:
    """Per-item cost and non-parallel overhead versus processors (cap=1e7).

    Left:  mean per-item time of KnapsackDP::ComputeItem and
           KnapsackDP::ComputeThreadBlock, i.e. K_s(p)/count_s with count_s=50.
    Right: overhead = ComputeItem_mean(p) - ComputeThreadBlock_mean(p)
           (the per-item work that is not inside the parallel thread block).
    """
    procs = sorted(df["processors"].unique())
    sub = df[df["family"] == 10]

    item_mean = [
        sub[(sub["processors"] == p) & (sub["capacity"] == 10_000_000)][
            "KnapsackDP_ComputeItem_mean"
        ].mean()
        for p in procs
    ]
    tb_mean = [
        sub[(sub["processors"] == p) & (sub["capacity"] == 10_000_000)][
            "KnapsackDP_ComputeThreadBlock_mean"
        ].mean()
        for p in procs
    ]
    overhead = np.array(item_mean) - np.array(tb_mean)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].plot(procs, item_mean, "o-", color="#1f77b4", label="ComputeItem mean")
    axes[0].plot(procs, tb_mean, "s--", color="#d62728", label="ComputeThreadBlock mean")
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xticks(procs)
    axes[0].set_xticklabels([str(p) for p in procs])
    axes[0].set_xlabel("processors")
    axes[0].set_ylabel("per-item time (ms, log)")
    axes[0].set_title("Per-item cost vs processors (max_weight=10, cap=1e7)")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()

    axes[1].plot(procs, overhead, "o-", color="#9467bd")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(procs)
    axes[1].set_xticklabels([str(p) for p in procs])
    axes[1].set_xlabel("processors")
    axes[1].set_ylabel("ComputeItem - ComputeThreadBlock (ms)")
    axes[1].set_title("Per-item non-parallel overhead")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_compute_speedup(metrics: pd.DataFrame, out: Path) -> None:
    """Wall-clock speedup vs speedup measured from the Compute time section.

        S(p)   = T(1) / T(p)          wall speedup
        S_K(p) = K(1) / K(p)          kernel speedup, K = KnapsackDP::Compute

    Relationship:  S(p) = S_K(p) * phi(p)/phi(1), where phi(p) is the Compute
    fraction of wall time, so the gap between the two curves is the fixed /
    serial overhead not included in the Compute section.
    """
    sub = metrics[metrics["family"] == 10]
    caps = sorted(sub["capacity"].unique())
    procs = sorted(sub["processors"].unique())

    fig, axes = plt.subplots(1, len(caps), figsize=(5.5 * len(caps), 5), sharey=True)
    if len(caps) == 1:
        axes = [axes]
    for ax, cap in zip(axes, caps):
        curve = sub[sub["capacity"] == cap].sort_values("processors")
        ax.plot(curve["processors"], curve["speedup"], "o-", label="wall speedup")
        ax.plot(curve["processors"], curve["compute_speedup"], "s--",
                label="Compute-section speedup")
        ax.plot(procs, procs, "k:", linewidth=1, label="ideal")
        ax.set_xscale("log", base=2)
        ax.set_xticks(procs)
        ax.set_xticklabels([str(p) for p in procs])
        ax.set_xlabel("processors (threads)")
        ax.set_title(f"capacity={cap:,}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("speedup")
    fig.suptitle("Wall vs Compute-section speedup (max_weight=10)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_relative_sections(metrics: pd.DataFrame, out: Path) -> None:
    """Relative section time per thread and the Compute share of wall time.

    Left:  R_s(p) = K_s(p) / K_s(1) for s in {Compute, ComputeItem,
           ComputeThreadBlock}; R_s(p) = 1.0 is the serial value and the ideal
           perfectly-scaling reference is R_ideal(p) = 1/p.
    Right: phi(p)   = K(p) / (T(p)*1000)              (Compute share, %)
           1 - phi(p) = overhead(p) / (T(p)*1000)     (setup/teardown, %)
    """
    sub = metrics[metrics["family"] == 10]
    procs = sorted(sub["processors"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for col, label, color in (
        ("compute_rel", "KnapsackDP::Compute", "#1f77b4"),
        ("item_rel", "KnapsackDP::ComputeItem", "#ff7f0e"),
        ("tb_rel", "KnapsackDP::ComputeThreadBlock", "#2ca02c"),
    ):
        curve = sub.groupby("processors")[col].mean().reindex(procs)
        axes[0].plot(procs, curve.values, "o-", color=color, label=label)
    axes[0].plot(procs, [1.0 / p for p in procs], "k:", linewidth=1, label="ideal 1/p")
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xticks(procs)
    axes[0].set_xticklabels([str(p) for p in procs])
    axes[0].set_xlabel("processors (threads)")
    axes[0].set_ylabel("relative section time  (1.0 = serial)")
    axes[0].set_title("Relative time sections per thread (max_weight=10, all caps)")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8)

    frac = sub.groupby("processors")["compute_fraction"].mean().reindex(procs)
    axes[1].plot(procs, frac.values * 100.0, "o-", color="#d62728",
                 label="Compute share of wall time")
    axes[1].plot(procs, (1.0 - frac.values) * 100.0, "s--", color="#9467bd",
                 label="setup/teardown + serial overhead")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(procs)
    axes[1].set_xticklabels([str(p) for p in procs])
    axes[1].set_xlabel("processors (threads)")
    axes[1].set_ylabel("fraction of wall time (%)")
    axes[1].set_ylim(0, 100)
    axes[1].set_title("Where the wall time goes as threads increase")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def print_weight_sections(metrics: pd.DataFrame, df: pd.DataFrame) -> None:
    """Weight-distribution effect measured through the time sections (serial, p=1).

    Reported quantities per weight family w and capacity c:

        K(w, c, 1)                     Compute section time (ms)
        mean/min/max_item(w, c, 1)     per-item ComputeItem statistics (ms, 50 items)
        I(w, c, 1) = max_item / mean_item          load imbalance
        R(w, c, 1) = K(w, c, 1) / K(10, c, 1)      relative cost vs max_weight=10
        R_max(w, c, 1) = max_item(w) / max_item(10) tail growth
    """
    serial = metrics[metrics["processors"] == 1].copy()

    print("\n=== Weight effect via sections (serial, p=1) ===")
    rows = []
    for cap in sorted(serial["capacity"].unique()):
        for family in FAMILY_ORDER:
            m = serial[(serial["capacity"] == cap) & (serial["family"] == family)].iloc[0]
            items = df[(df["family"] == family) & (df["capacity"] == cap)
                       & (df["processors"] == 1)]
            rows.append({
                "capacity": cap,
                "max_weight": family,
                "compute_ms": m["compute_ms"],
                "item_mean_ms": items["KnapsackDP_ComputeItem_mean"].mean(),
                "item_min_ms": items["KnapsackDP_ComputeItem_min"].mean(),
                "item_max_ms": items["KnapsackDP_ComputeItem_max"].mean(),
                "item_var": items["KnapsackDP_ComputeItem_variance"].mean(),
                "load_imbalance": items["KnapsackDP_ComputeItem_max"].mean()
                / items["KnapsackDP_ComputeItem_mean"].mean(),
            })
    table = pd.DataFrame(rows)
    base = table[table["max_weight"] == 10].set_index("capacity")
    table["compute_rel"] = table.apply(
        lambda r: r["compute_ms"] / base.loc[r["capacity"], "compute_ms"], axis=1
    )
    table["item_max_rel"] = table.apply(
        lambda r: r["item_max_ms"] / base.loc[r["capacity"], "item_max_ms"], axis=1
    )
    print(table.round(4).to_string(index=False))

    print("\n=== Overhead of the widest weight range (max_weight=1000 vs 10) ===")
    pivot = table.pivot(index="capacity", columns="max_weight", values=["compute_ms", "load_imbalance"])
    print(pivot.round(4).to_string())


def plot_weight_sections(metrics: pd.DataFrame, df: pd.DataFrame, out: Path) -> None:
    """Weight-distribution effect seen through the time sections.

    Left:   K_w(c, 1) vs capacity, one line per weight family w.
    Middle: per-item min/mean/max at c=1e7, p=1; bars = mean_item(w),
            error bars span [min_item(w), max_item(w)] over the 50 items.
    Right:  load imbalance I(w, c, 1) = max_item(w, c, 1) / mean_item(w, c, 1).
    """
    serial = metrics[metrics["processors"] == 1]
    caps = sorted(serial["capacity"].unique())
    procs = sorted(df["processors"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    for family in FAMILY_ORDER:
        curve = serial[serial["family"] == family].set_index("capacity").reindex(caps)
        axes[0].plot(caps, curve["compute_ms"], "o-", color=FAMILY_COLORS[family],
                     label=f"max_weight={family}")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("capacity (log)")
    axes[0].set_ylabel("Compute section time, serial (ms, log)")
    axes[0].set_title("Kernel cost vs capacity per weight range")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()

    cap = 10_000_000
    width = 0.25
    positions = np.arange(len(FAMILY_ORDER))
    item_mean, item_lo, item_hi = [], [], []
    for family in FAMILY_ORDER:
        items = df[(df["family"] == family) & (df["capacity"] == cap) & (df["processors"] == 1)]
        mean = items["KnapsackDP_ComputeItem_mean"].mean()
        item_mean.append(mean)
        item_lo.append(mean - items["KnapsackDP_ComputeItem_min"].mean())
        item_hi.append(items["KnapsackDP_ComputeItem_max"].mean() - mean)
    axes[1].bar(positions, item_mean, width * 2, color=[FAMILY_COLORS[f] for f in FAMILY_ORDER])
    axes[1].errorbar(positions, item_mean, yerr=[item_lo, item_hi], fmt="none",
                     ecolor="black", capsize=6)
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels([f"max_weight={f}" for f in FAMILY_ORDER])
    axes[1].set_ylabel("per-item time, serial (ms)")
    axes[1].set_title(f"Per-item min/mean/max (capacity={cap:,}, p=1)")
    for pos, mean in zip(positions, item_mean):
        axes[1].text(pos, mean, f"{mean:.1f}ms", ha="center", va="bottom", fontsize=9)
    axes[1].grid(True, axis="y", alpha=0.3)

    for family in FAMILY_ORDER:
        rows = []
        for cap in caps:
            items = df[(df["family"] == family) & (df["capacity"] == cap) & (df["processors"] == 1)]
            rows.append(
                items["KnapsackDP_ComputeItem_max"].mean()
                / items["KnapsackDP_ComputeItem_mean"].mean()
            )
        axes[2].plot(caps, rows, "o-", color=FAMILY_COLORS[family], label=f"max_weight={family}")
    axes[2].set_xscale("log")
    axes[2].set_xlabel("capacity (log)")
    axes[2].set_ylabel("per-item max / mean")
    axes[2].set_title("Load imbalance of the 50 items (serial)")
    axes[2].grid(True, which="both", alpha=0.3)
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    metrics = add_metrics(df)
    print_summary(metrics)
    print_weight_sections(metrics, df)

    plots = {
        "kpdp_speedup_vs_processors.png": lambda: plot_speedup_vs_processors(metrics, OUT_DIR / "kpdp_speedup_vs_processors.png"),
        "kpdp_compute_speedup.png": lambda: plot_compute_speedup(metrics, OUT_DIR / "kpdp_compute_speedup.png"),
        "kpdp_relative_sections.png": lambda: plot_relative_sections(metrics, OUT_DIR / "kpdp_relative_sections.png"),
        "kpdp_efficiency_vs_processors.png": lambda: plot_efficiency_vs_processors(metrics, OUT_DIR / "kpdp_efficiency_vs_processors.png"),
        "kpdp_speedup_heatmap.png": lambda: plot_speedup_heatmap(metrics, OUT_DIR / "kpdp_speedup_heatmap.png"),
        "kpdp_time_vs_processors.png": lambda: plot_time_vs_processors(metrics, OUT_DIR / "kpdp_time_vs_processors.png"),
        "kpdp_capacity_effect.png": lambda: plot_capacity_effect(metrics, OUT_DIR / "kpdp_capacity_effect.png"),
        "kpdp_weight_effect.png": lambda: plot_weight_effect(metrics, OUT_DIR / "kpdp_weight_effect.png"),
        "kpdp_weight_sections.png": lambda: plot_weight_sections(metrics, df, OUT_DIR / "kpdp_weight_sections.png"),
        "kpdp_repeatability.png": lambda: plot_repeatability(df, OUT_DIR / "kpdp_repeatability.png"),
        "kpdp_section_breakdown.png": lambda: plot_section_breakdown(metrics, df, OUT_DIR / "kpdp_section_breakdown.png"),
    }
    for name, func in plots.items():
        func()
        print(f"saved {OUT_DIR / name}")


if __name__ == "__main__":
    main()
