"""Analysis of the GPU (CUDA) knapsack DP block-size sweep.

Data source: doc/data/perf/perf_kpdpgpu.csv produced by sbatch/run_gpu.sbatch.
For each (max_weight, capacity, threads_per_block) it holds 3 repeats with
  time_s  solver wall time from kpbench stdout (includes CUDA setup,
          allocations, transfers, backtrack),
  dp_ms   host-measured time of the DP kernel loop only (n sequential
          kernel launches, one per DP row).

The sweep varies threads per block t in {32,64,128,256,512,1024}; the grid
always holds ceil((C+1)/t) blocks, i.e. one thread per DP cell: total
parallelism is fixed, t only changes the launch geometry and occupancy.

Plots are written to doc/images/. Figures:
  perf_kpdpgpu_block_size.png  dp_ms and speedup S(32)/S(t) vs t
  perf_kpdpgpu_vs_cpu.png      GPU vs CPU (serial / 32-thread) speedup
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
GPU_CSV = BASE_DIR / "data" / "perf" / "perf_kpdpgpu.csv"
CPU_CSV = BASE_DIR / "data" / "perf" / "perf_kpdp.csv"
OUT_DIR = BASE_DIR / "images"

NUM_ITEMS = 50  # every benchmark run uses num_items=50 (see sbatch scripts)

CAP_COLORS = {100_000: "#1f77b4", 1_000_000: "#2ca02c", 10_000_000: "#d62728"}
FAMILY_COLORS = {10: "#1f77b4", 1000: "#d62728"}
THREADS_ORDER = [32, 64, 128, 256, 512, 1024]


def median_table(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["max_weight", "capacity", "threads"], as_index=False)
        [["dp_ms", "time_s"]]
        .median()
    )


def print_tables(g: pd.DataFrame) -> None:
    for family in sorted(g["max_weight"].unique()):
        sub = g[g["max_weight"] == family]
        print(f"\n== dp_ms median, max_weight={family} ==")
        print(sub.pivot(index="capacity", columns="threads", values="dp_ms").round(2).to_string())
    print("\n== time_s median (whole solver incl. CUDA setup) ==")
    print(g.pivot(index=["max_weight", "capacity"], columns="threads", values="time_s").round(3).to_string())

    # Overhead floor and per-cell incremental cost at the best block size:
    # dp_ms(cap=1e5/1e6) is flat (floor: 50 kernel launches + sync), the
    # slope between 1e6 and 1e7 isolates the bandwidth-limited per-cell cost.
    print("\n== overhead floor and incremental per-cell cost (best block size) ==")
    for family in sorted(g["max_weight"].unique()):
        sub = g[g["max_weight"] == family]
        best = sub.loc[sub.groupby("capacity")["dp_ms"].idxmin()].set_index("capacity")
        floor = best.loc[best.index < 10_000_000, "dp_ms"].median()
        slope = (best.loc[10_000_000, "dp_ms"] - best.loc[1_000_000, "dp_ms"]) / (
            (NUM_ITEMS + 1) * (10_000_000 - 1_000_000)
        )
        print(f"max_weight={family}: floor ~{floor:.2f} ms, "
              f"incremental cost ~{slope * 1e9:.2f} ns/cell "
              f"(best block sizes: {best['threads'].tolist()})")


def plot_block_size(g: pd.DataFrame, out: Path) -> None:
    families = sorted(g["max_weight"].unique())
    caps = sorted(g["capacity"].unique())
    fig, axes = plt.subplots(2, len(families), figsize=(6.2 * len(families), 8), sharex=True)

    for j, family in enumerate(families):
        sub = g[g["max_weight"] == family]
        ax_t, ax_s = axes[0][j], axes[1][j]
        for cap in caps:
            curve = sub[sub["capacity"] == cap].set_index("threads").reindex(THREADS_ORDER)
            ax_t.plot(THREADS_ORDER, curve["dp_ms"], "o-",
                      color=CAP_COLORS[cap], label=f"capacity={cap:.0e}")
            speedup = curve["dp_ms"].iloc[0] / curve["dp_ms"]
            ax_s.plot(THREADS_ORDER, speedup, "o-",
                      color=CAP_COLORS[cap], label=f"capacity={cap:.0e}")
        for ax, ylab, title in ((ax_t, "kernel time dp_ms (median)", f"max_weight={family}"),
                                (ax_s, r"speedup $S = t(32)/t(t)$", "")):
            ax.set_xscale("log", base=2)
            ax.set_xticks(THREADS_ORDER, [str(t) for t in THREADS_ORDER])
            ax.grid(True, which="both", alpha=0.3)
            ax.set_xlabel("threads per block")
            if ylab:
                ax.set_ylabel(ylab)
            if title:
                ax.set_title(title)
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_vs_cpu(g: pd.DataFrame, cpu: pd.DataFrame, out: Path) -> None:
    """GPU speedup over the CPU classic version at matching configurations.

    cpu must come from perf_kpdp.csv (same seed/num_items, capacities 1e6/1e7).
    Two panels: wall-to-wall speedup (GPU time_s includes CUDA setup, which
    dominates for small tables) and kernel-only speedup (GPU dp_ms against
    the CPU wall time, which for the CPU version is essentially all compute).
    GPU block size is the best per configuration.
    """
    caps = sorted(set(g["capacity"]) & set(cpu["capacity"]))
    families = sorted(set(g["max_weight"]) & set(cpu["max_weight"]))
    rows = []
    for family in families:
        for cap in caps:
            gpu_sub = g[(g["max_weight"] == family) & (g["capacity"] == cap)]
            gpu_best = gpu_sub["time_s"].min()
            gpu_best_k = gpu_sub.loc[gpu_sub["time_s"].idxmin(), "dp_ms"]
            cpu_sub = cpu[(cpu["max_weight"] == family) & (cpu["capacity"] == cap)]
            for cpu_threads, label in ((1, "CPU serial"), (32, "CPU 32 threads")):
                cpu_t = cpu_sub[cpu_sub["threads"] == cpu_threads]["time_s"].median()
                if np.isfinite(cpu_t) and gpu_best > 0:
                    rows.append({"max_weight": family, "capacity": cap, "baseline": label,
                                 "wall_speedup": cpu_t / gpu_best,
                                 "kernel_speedup": cpu_t * 1000.0 / gpu_best_k})
    cmp_df = pd.DataFrame(rows)
    if cmp_df.empty:
        print("no overlapping configurations for GPU vs CPU comparison")
        return
    print("\n== GPU vs CPU (GPU best block size per config) ==")
    print(cmp_df.round(3).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    markers = {"CPU serial": "o", "CPU 32 threads": "s"}
    for j, (col, title) in enumerate((("wall_speedup", "wall time (incl. CUDA setup)"),
                                      ("kernel_speedup", "GPU kernel only vs CPU wall"))):
        for (family, baseline), sub in cmp_df.groupby(["max_weight", "baseline"]):
            sub = sub.sort_values("capacity")
            axes[j].plot(sub["capacity"], sub[col], markers[baseline] + "-",
                         color=FAMILY_COLORS[family],
                         label=f"vs {baseline}, max_weight={family}")
        axes[j].set_xscale("log")
        axes[j].set_yscale("log")
        axes[j].axhline(1.0, color="k", linestyle=":", linewidth=1)
        axes[j].set_xlabel("capacity (log)")
        axes[j].set_title(title)
        axes[j].grid(True, which="both", alpha=0.3)
        axes[j].legend(fontsize=8)
    axes[0].set_ylabel(r"speedup $T_{cpu}/T_{gpu}$")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    gpu = pd.read_csv(GPU_CSV)
    g = median_table(gpu)
    print_tables(g)
    plot_block_size(g, OUT_DIR / "perf_kpdpgpu_block_size.png")
    print(f"saved {OUT_DIR / 'perf_kpdpgpu_block_size.png'}")

    cpu = pd.read_csv(CPU_CSV)
    cpu = cpu[cpu["event"] == "cycles"] if "event" in cpu.columns else cpu
    cpu = (
        cpu.groupby(["max_weight", "capacity", "threads"], as_index=False)["time_s"]
        .median()
    )
    plot_vs_cpu(g, cpu, OUT_DIR / "perf_kpdpgpu_vs_cpu.png")
    print(f"saved {OUT_DIR / 'perf_kpdpgpu_vs_cpu.png'}")


if __name__ == "__main__":
    main()
