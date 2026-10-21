"""Speedup-vs-threads curves of the classic KnapsackDP with the item count
as the scaling variable -- the counterpart of kpdpdag_n_speedup.py.

Data: doc/data/perf/perf_kpdp_n.csv from sbatch/run_kpdp_n.sbatch
(n {50,250,1000,5000} x threads {1,2,4,8,16,32} x 2 repeats, capacity 1e6,
seed 42).  S(p) = T(1)/T(p) per n, medians over the repeats.

Output: doc/images/kpdp_speedup_vs_n.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
CSV = BASE_DIR / "data" / "perf" / "perf_kpdp_n.csv"
OUT = BASE_DIR / "images" / "kpdp_speedup_vs_n.png"

N_COLORS = {50: "#1f77b4", 250: "#ff7f0e", 1000: "#2ca02c", 5000: "#d62728"}


def main() -> None:
    df = pd.read_csv(CSV)
    g = df.groupby(["n", "threads"], as_index=False)["time_s"].median()

    fig, ax = plt.subplots(figsize=(8, 5))
    for n, sub in g.groupby("n"):
        sub = sub.sort_values("threads")
        t1 = sub[sub["threads"] == 1]["time_s"].iloc[0]
        ax.plot(sub["threads"], t1 / sub["time_s"], "o-", color=N_COLORS[n], label=f"n={n}")
    ax.plot([1, 32], [1, 32], "k--", linewidth=1, label="ideale")

    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32], ["1", "2", "4", "8", "16", "32"])
    ax.set_xlabel("thread")
    ax.set_ylabel(r"speedup a muro $S = T(1)\/T(p)$")
    ax.set_title("KnapsackDP: speedup al variare di thread e numero di oggetti\n"
                 "(capacity $10^6$, mediana su 2 repliche)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
