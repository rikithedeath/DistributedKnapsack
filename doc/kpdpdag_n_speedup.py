"""Speedup-vs-threads curves of KnapsackDPDAG with the item count as the
scaling variable.

Data: doc/data/perf/perf_kpdpdag_n.csv from sbatch/run_kpdpdag_n.sbatch
(n {50,250,1000,5000} x threads {1,2,4,8,16,32} x tile config
{(25,1e4),(5,1e4),(25,1e3),(5,1e3)} x 2 repeats, capacity 1e6, seed 42).
For every (n, p) the fastest of the four tile granularities is used; S(p) = T(1)/T(p) with
T(1) taken from the best serial run of the same n.

Output: doc/images/perf_kpdpdag_speedup_vs_n.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
CSV = BASE_DIR / "data" / "perf" / "perf_kpdpdag_n.csv"
OUT = BASE_DIR / "images" / "perf_kpdpdag_speedup_vs_n.png"

N_COLORS = {50: "#1f77b4", 250: "#ff7f0e", 1000: "#2ca02c", 5000: "#d62728"}


def main() -> None:
    df = pd.read_csv(CSV)
    g = df.groupby(["n", "threads", "item_block", "cap_block"], as_index=False)["time_s"].median()

    fig, ax = plt.subplots(figsize=(8, 5))
    for n, sub in g.groupby("n"):
        # best tile granularity per thread count
        best = sub.loc[sub.groupby("threads")["time_s"].idxmin()].sort_values("threads")
        t1 = best[best["threads"] == 1]["time_s"].iloc[0]
        speedup = t1 / best["time_s"]
        ax.plot(best["threads"], speedup, "o-", color=N_COLORS[n], label=f"n={n}")
    ax.plot([1, 32], [1, 32], "k--", linewidth=1, label="ideal")

    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32], ["1", "2", "4", "8", "16", "32"])
    ax.set_xlabel("thread")
    ax.set_ylabel("speedup")
    ax.set_ylim(0, 10)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
