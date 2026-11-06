# Performance-Counter Analysis — Measures and Graphs Explained

> ⚠️ **Warning about the classic-DP measurements.** From the commit "complete
> knapsackdp granularization" (2026-09-15) until the "Benchmark fairness:
> row-major classic DP" revision, the classic DP used an `Eigen::MatrixX<int>`,
> which Eigen stores **column-major**: with the inner loop over the capacity,
> consecutive cells were `(n+1)*4` bytes apart. The classic-DP timings and
> counters analysed here measure that layout, not the algorithm, and every
> DAG-vs-classic comparison built on them has to be redone
> (`sbatch/run_timing.sbatch`). The updated report is `doc/main.typ`.

This document explains, measure by measure and graph by graph, the analysis
produced by `doc/kpdp_perf_analysis.py`. That script turns raw `perf stat`
hardware counters (in `doc/data/perf/perf_summary.csv` and
`doc/data/perf/perf_summary_dag.csv`) into metrics, tables and the nine
`doc/images/perf_*.png` / `perf_vs_algorithms.png` figures, and pairs them
with the clean wall-clock timings of `doc/data/result_kpdp.json` and
`doc/data/result_kpdpdag.json`.

The two algorithms analysed are:

- **KnapsackDP** — the classic dynamic-programming kernel over the
  `(num_items+1) × (capacity+1)` table, parallelised with OpenMP over
  capacity chunks.
- **KnapsackDPDAG** — the *tiled* variant (`src/Knapsack/knapsackdpdag.cpp`):
  the same table is partitioned into tiles of `item_block` items ×
  `cap_block` capacity slots, executed as a dependency DAG of tasks. Its
  behaviour is tuned by the two block parameters.

Reproduce everything with: `~/pyenv/sci/bin/python doc/kpdp_perf_analysis.py`.

---

## 1. Where the numbers come from

| Source | Content | Grid covered |
|---|---|---|
| `perf_summary.csv` | 12 runs × 11 counter events | KnapsackDP: max_weight {10, 1000} × capacity {1e6, 1e7} × threads {1, 8, 32} |
| `perf_summary_dag.csv` | 360 runs × 11 counter events | KnapsackDPDAG: max_weight {10, 100, 1000} × capacity {1e4, 1e5, 1e6} × threads {1–32} × item_block {5, 10, 15, 25} × cap_block {1000, 10000} (cap_block ≥ capacity skipped) |
| `result_kpdp.json` | clean wall times (host epito02) | superset of the KnapsackDP counter grid |
| `result_kpdpdag.json` | clean wall times (host epito01); also embeds plain `knapsackdp` baselines on epito01 | superset of the DAG counter grid |

Three properties of this data shape every interpretation below:

1. **Wall times never come from the perf runs.** `perf stat` slows a program
   down; the CSVs only carry hardware counters (plus `runtime_ns`, the time a
   counter group was scheduled on the PMU, which is *not* a wall time). Timings
   are therefore joined from the clean benchmark JSONs.
2. **Coverage is low and reported, not filtered.** The CPU offers fewer
   counters than the 11 events we request, so the kernel time-shares them
   (multiplexing). `coverage_pct` is the fraction of time the counters were
   actually measuring; values are statistically scaled estimates. Our runs sit
   between 33 % and 56 % coverage — ratios are trustworthy in trend, not to
   three decimals. Every table carries `coverage_pct` and the script prints a
   summary per file; nothing is dropped (report-only decision).
3. **The two timing JSONs come from different hosts** (epito02 for plain DP,
   epito01 for the DAG). Counter *ratios* are within-run, so they are
   unaffected; for the cross-algorithm *time* comparison the script uses the
   plain-DP baselines embedded in `result_kpdpdag.json`, so both algorithms are
   timed on epito01.

All problem instances share `num_items = 50`, `seed = 42`, `min_weight = 1`.

---

## 2. The raw counters

A hardware performance counter is a register inside the CPU that increments
every time a specific micro-architectural event happens. `perf stat` reads
them around one program run. Our 11 events:

| Event | Meaning |
|---|---|
| `cycles` | CPU clock ticks consumed. **Counted over all cores that executed the program's threads** — with 32 threads, 32 cores add up. This is why "total cycles" usually *grows* when adding threads. |
| `instructions` | Machine instructions that retired (finished). |
| `L1-dcache-loads` / `L1-dcache-load-misses` | Data loads served by the per-core first-level cache (~64 KB, a few cycles) / how many had to be fetched from further out (~20–40 cycles). |
| `cache-references` / `cache-misses` | Accesses that reached the last-level cache / missed it and went to DRAM (~100+ cycles). |
| `ll_cache_rd` / `ll_cache_miss_rd` | The node's mesh/last-level-cache accesses and misses (ARM naming). This is the LLC statistic used in the analysis. |
| `dTLB-load-misses` / `iTLB-load-misses` | Data / instruction address-translation lookups that missed the TLB — relevant for huge tables that span many memory pages. |
| `branch-misses` | Speculative branch predictions that guessed wrong and had to be rolled back. |

`perf` also emits a few *derived* rows (e.g. "insn per cycle"). The script
recomputes all ratios from the raw values instead — perf's derived numbers are
rounded to two decimals — and cross-checks them against perf's own values
(36 + 1080 checks, 0 mismatches beyond rounding).

---

## 3. The derived measures

Notation: `w` = weight family (`max_weight`), `c` = capacity, `p` = threads,
`ev(w,c,p)` = raw value of event `ev` for that configuration.

### IPC — instructions per cycle

```
IPC(p) = instructions / cycles
```

How much work the CPU extracts from every clock tick. Modern cores can
complete several instructions per cycle (IPC > 2 is achievable). An IPC well
below 1 means the pipeline is *stalled*: the core has clock ticks available
but no ready instructions — almost always because it is waiting for memory.
IPC is the single best "compute-bound or memory-bound?" indicator in this
dataset.

### MPKI — misses per kilo-instruction

```
MPKI_e(p) = miss_e / instructions × 1000     e ∈ {L1, LLC, dTLB}
```

Miss counts alone are meaningless when comparing runs of different length and
capacity; normalising by retired instructions makes a 0.07 s run and a 6 s run
comparable. "LLC MPKI = 69" reads: *for every 1000 instructions executed, 69
loads had to go all the way to DRAM.* Values of a few MPKI are normal; values
of 50–100 mean the program is essentially streaming through memory.

### Hit ratio

```
hit_e(p) = (1 − miss_e / loads_e) × 100 %
```

Fraction of accesses the cache level could satisfy. A serialised form appears
in the tables; values near 0 % for the LLC mean the cache is effectively
useless for the access pattern.

### Branch-miss rate and dTLB MPKI

Secondary health indicators (reported in the tables, not plotted):
`B(p) = branch-misses / instructions × 100`, `MPKI_dTLB =
dTLB-load-misses / instructions × 1000`.

### Cycles per DP cell

```
cycles_per_cell = cycles / ((num_items + 1) × (capacity + 1))
```

The DP algorithm's elementary work is "fill one cell of the table". Dividing
total cycles by the number of cells gives the hardware cost of one cell — a
work-normalised efficiency measure. Crucially, **both algorithms fill exactly
the same table** (the DAG only tiles it), so this is the fairest
cross-algorithm comparison metric.

Warning: `cycles` sums over all cores. For `p > 1` the metric therefore counts
the *whole machine's* time per cell, including spinning and synchronisation —
it grows with p even if each cell got cheaper to compute. The serial (p = 1)
value is the "pure kernel" number; the p > 1 values measure total machine
effort per unit of work.

### Wall speedup and the counter decomposition

```
S_wall(p) = T(1) / T(p)              from the clean timing JSONs
S_cyc(p)  = cycles(1) / cycles(p)    total-cycle reduction (see warning below)
rate(p)   = cycles(p) / (T(p) × p)   cycles per core-second
r(p)      = rate(p) / rate(1)        per-core cycle-rate ratio
```

Because `cycles` sums over all cores, `S_cyc < 1` does **not** mean the
parallel run does fewer cycles per core — it means the machine burned more
total cycles (spin-waits, barriers, duplicated cache-coherence traffic). To
still get an exact accounting, define the per-core rate:

- `rate(p) = cycles(p) / (T(p) · p)` is how many cycles an *average active
  core* emitted per second of wall time.
- `r(p) = rate(p) / rate(1)` compares it to the serial core: it folds together
  the clock-frequency ratio and the fraction of time a core spends doing
  counted work vs. waiting.

The identity (exact by construction, up to the small timing skew between the
perf run and the clean timing):

```
S_wall(p) = T(1)/T(p) = [cycles(1)/cycles(p)] × p × r(p) = S_cyc × p × r
```

So the speedup factors into three questions: *did the total cycle count grow?
* (S_cyc), *how many cores were there?* (p), and *how fast did each core run?*
(r). The decomposition plots show exactly these three bars, with
`S_wall = S_cyc × p·r` holding by construction.

### Tiles and DAG configuration

```
tiles = ⌈num_items / item_block⌉ × ⌈(capacity + 1) / cap_block⌉
```

Number of tasks in the dependency DAG. `item_block` (rows) trades parallelism
against per-tile overhead; `cap_block` (columns) trades tile *count* against
tile *footprint*. The "best configuration" for a given (family, capacity,
threads) is the one with the smallest clean wall time; its serial baseline is
the best configuration at p = 1 of the same (family, capacity), so
`S_wall = t_best(1) / t_best(p)` compares "the algorithm at its best" at both
ends.

### Coverage (report-only)

`coverage_pct = min over the run's events of runtime_pct` — the worst
multiplexing quality among the 11 events of that run. It is printed and shown
in every table; no row is ever dropped.

---

## 4. The graphs

### 4.1 `perf_kpdp_ipc_vs_threads.png` — plain DP: per-core efficiency

**Plots:** one panel per weight family (10, 1000); x = threads (1, 8, 32);
y = IPC; one line per capacity (1e6, 1e7).

**How to read:** falling lines mean each instruction stream gets *less*
efficient as threads are added; the gap between the two capacities shows the
effect of table size.

**Results:** serial IPC is 0.62–0.75 (never reaching 1); at 8 threads
0.45–0.58; at 32 threads it collapses to **0.19–0.32**. The larger capacity is
always the lower line.

**Meaning:** even single-threaded, the DP inner loop stalls on memory loads —
the CPU completes less than one instruction per cycle. Adding threads does not
improve (and actively degrades) per-instruction efficiency: contention and
spin loops consume cycles that produce no instructions. The speedup of plain
DP comes from *more cores*, not better ones — and the next figures show the
memory system is the culprit.

### 4.2 `perf_kpdp_memory_intensity.png` — why the IPC is low

**Plots:** left = L1 MPKI, right = LLC MPKI; x = threads; colour = weight
family (blue 10, green 1000); solid = capacity 1e6, dashed = 1e7.

**How to read:** height = memory pressure per unit of work; two aspects
matter — the absolute level, and whether threads change it.

**Results:** L1 MPKI 23–55; LLC MPKI **54–98** with LLC hit ratios around
0 % (a few slightly negative values are multiplexing noise, see §6).
max_weight = 1000 sits consistently above max_weight = 10 (e.g. 98.5 vs 93.3
LLC MPKI at cap 1e7, serial). All curves decline mildly with threads.

**Meaning:** for every 1000 instructions, ~70–100 loads miss the last-level
cache entirely and go to DRAM. The table is ~200 MB (cap 1e6) to ~2 GB
(1e7) — orders of magnitude beyond any cache, so plain DP is a DRAM-streaming
kernel. The wider weight range is worse because a "take item i" step reads
`dp[i−1][c − w_i]`, up to w_i = 1000 columns back: the access pattern scatters
over a wider window of the previous row and touches more cache lines per
cell. The mild decline with threads reflects OpenMP capacity chunking (each
thread walks a smaller slice) and spin instructions diluting the denominator —
the pressure remains enormous.

### 4.3 `perf_kpdp_cycles_per_cell.png` — machine cost per unit of work

**Plots:** one panel per family; x = capacity (log–log); y = total cycles per
DP cell; one line per thread count (1, 8, 32).

**How to read:** the serial line is the "pure" kernel cost per cell; the p > 1
lines count *all* cores' cycles, so their height above the serial line measures
total machine effort (useful work + spinning + coordination) per unit of work.

**Results:** serial falls from ≈ 47 (cap 1e6) to ≈ 41 (cap 1e7) cycles/cell;
the 8-thread line sits at ≈ 57–64; the 32-thread line at ≈ **139–150** —
independent of family.

**Meaning:** (a) per-cell cost slightly *improves* with capacity — fixed
per-item/per-row overheads amortise over more cells and long contiguous rows
stream well; (b) at 32 threads the machine spends ~3× more total cycles per
computed cell than serial. This is the cycle inflation quantified by
`S_cyc ≈ 0.27–0.34` in the decomposition: parallelism does not reduce the
per-cell hardware cost, it multiplies the machine-time spent per cell.

### 4.4 `perf_kpdp_decomposition.png` — where the 32-thread speedup went

**Plots:** one panel per family; x = capacity; three bars per capacity:
**S_wall** (blue, achieved speedup), **S_cyc** (orange, total-cycle change),
**p·r** (green, cores × per-core rate). All at 32 threads; by construction
`S_wall = S_cyc × p·r`.

**How to read:** the gap between 32 (ideal) and S_wall is explained by the
other two bars: orange < 1 = wasted total cycles; green < 32 = cores not
running at full serial rate.

**Results:** S_wall ≈ 7.1–7.5; S_cyc ≈ 0.27–0.34 (cycles *tripled*);
p·r ≈ 21.9–27.3, i.e. r ≈ 0.68–0.85 per core.

**Meaning:** of the ideal 32×, a factor ≈ 3 is lost to extra total cycles
(spinning/synchronisation burning cycles that compute nothing extra) and a
factor ≈ 4 to cores running at 68–85 % of their serial effective rate
(stalls at memory and barriers). The product recovers the measured ≈ 7.4.
This is the quantitative answer to "why doesn't plain DP scale past ~7×?".

### 4.5 `perf_kpdpdag_block_grid.png` — the tiling knobs, dissected

**Plots:** three heatmaps over item_block (rows, 25…5 top to bottom) ×
cap_block (columns: 1000, 10000) at one representative point (max_weight=10,
capacity=1e5, 8 threads): clean wall time (log colour), IPC, LLC MPKI. Cell
values are printed.

**How to read:** the three panels together separate *scheduling cost* from
*cache quality*: if time tracked MPKI, tiling would be a locality story; if
time tracked tile count while MPKI stays flat, it is an overhead story.

**Results:** time: cap_block = 1000 costs 0.056–0.065 s vs 0.013–0.018 s for
cap_block = 10000 (3–4× apart). IPC: the *ranking inverts* — small cap_block
has the highest IPC (1.53–1.72) vs 1.28–1.48 for large. LLC MPKI: nearly flat
(3.2–6.8) everywhere. Also visible: at cap_block = 10000, item_block = 25 is
the *slowest* large-tile choice (0.0184 s).

**Meaning:** within this sweep, cache behaviour is decent for *all* block
sizes (MPKI 3–7 vs 69+ for plain DP); the 3–4× time gap is **not** a locality
effect — small cap_block fragments the table into ~5–45× more tiles
(e.g. 202–1010 vs 22–110 tiles here) and drowns in per-tile scheduling
overhead, even though each tile executes more efficiently (higher IPC). The
item_block = 25 anomaly shows the other edge of the trade-off: 2 × 11 = 22
tiles barely feed 8 threads, so the best configuration at this point is
item_block = 10 (55 tiles). Block choice is a balance between scheduling
overhead (few tiles) and parallelism (enough tiles).

### 4.6 `perf_kpdpdag_miss_vs_tiles.png` — locality or overhead?

**Plots:** 2 × 2 grid at p = 1, max_weight = 10. Top row: LLC MPKI vs number
of tiles; bottom row: wall time vs tiles. Columns: capacity 1e5, 1e6. Colour =
cap_block (blue 1000, red 10000), marker = item_block (○ 5, □ 10, △ 15, ◇ 25).
Log–log axes.

**How to read:** a downward-sloping top row would mean "more tiles → fewer
misses" (locality story); a flat top row with a sloping bottom row means the
time differences come from somewhere else (overhead story).

**Results:** the top row is nearly flat: LLC MPKI stays within 6.3–7.0
(cap 1e5) and 4.6–5.2 (cap 1e6) while the tile count spans ~22 to ~10,000.
The bottom row varies 3–4× (e.g. 0.06 s → 0.014 s at cap 1e5), with red
(cap_block = 10000, few tiles) fastest.

**Meaning:** at these block sizes the DAG's serial advantage is **not**
produced by better cache behaviour — misses per instruction are roughly
constant — but by executing far fewer DAG scheduling events. The cache win
versus plain DP (§4.9) comes from tiling *at all*, not from tiling *finer*.
This is a key, slightly counterintuitive finding for the paper: within the
swept range, choose few tiles.

### 4.7 `perf_kpdpdag_ipc_vs_threads.png` — the tiled kernel's efficiency

**Plots:** one panel per family (10, 100, 1000); x = threads 1–32; y = IPC of
the per-thread-count *best* configuration; one line per capacity.

**How to read:** compare the level against plain DP (0.19–0.75) and note which
capacity is on top.

**Results:** IPC ranges 1.13 → **2.44**, rising with thread count; the largest
capacity (1e6) has the *highest* IPC — the opposite ordering of plain DP. The
winning configuration often changes with p (e.g. at cap 1e6 the best config
moves from item_block 10 to 5 beyond 4 threads, growing tiles 505 → 1010).

**Meaning:** tiling makes each tile's working set (ib × cb cells) small enough
to stay cache-resident, so the relevant footprint is the tile, not the
200 MB table — which is why the biggest table is no longer the slowest per
instruction. More threads also overlap the DAG's wavefront execution better,
pushing IPC above 2: the kernel becomes genuinely compute-bound.

### 4.8 `perf_kpdpdag_decomposition.png` — the DAG's parallel ceiling

**Plots:** same three-bar decomposition as §4.4 (S_wall, S_cyc, p·r at 32
threads), for the best configuration, panels = family, groups = capacity.

**How to read:** as §4.4, but here the interesting bar is blue: how much
speedup 32 threads add on top of an already cache-optimal serial kernel.

**Results:** S_wall ≈ **0.96–1.22** — adding 31 threads changes almost
nothing. S_cyc ≈ 0.35–0.94 (total cycles still grow); p·r ≈ 1.2–3.4.

**Meaning:** the DAG's advantage is *algorithmic*, not a parallel-scaling
story. Its best serial configuration is so fast (~0.07 s at cap 1e6) and
already so cache-friendly that 32 threads have little left to harvest, while
coordination of thousands of tiles adds cycles (S_cyc < 1) almost matching the
cores' contribution. Plain DP reaches a much higher *relative* speedup
(7.4×) — but from a 9.3× slower starting point, so the DAG still wins in
absolute time at every thread count (§4.9).

### 4.9 `perf_vs_algorithms.png` — the headline comparison

**Plots:** rows = family (10, 1000); columns = IPC, cycles per DP cell, LLC
MPKI; x = threads {1, 8, 32}; blue = plain KnapsackDP, red = best DAG
configuration; capacity = 1e6 (the only capacity covered by both counter
sweeps). Timings for both algorithms come from epito01 (the plain-DP baselines
embedded in `result_kpdpdag.json`).

**How to read:** for each metric, the height ratio of the two bars is the
hardware-level advantage of one algorithm over the other at identical work
(same DP table, same cells).

**Results (p = 1):**

| Metric | plain | DAG best | ratio |
|---|---|---|---|
| IPC | 0.75 | 1.75 | 2.3× |
| cycles per DP cell | 47.2 | 20.1 | 2.3× fewer |
| LLC MPKI | 68.7 | **4.75** | **14.5× fewer** |
| wall time (same host) | 0.648 s | 0.070 s | **9.3× faster** |

At 32 threads: IPC 0.32 vs 2.40; cycles/cell 139 vs 59; MPKI 54.5 vs 1.44;
time 0.087 s vs 0.063 s (DAG 1.4× faster; time ratio 0.73 vs 0.11 at p = 1).

**Meaning:** tiling converts a DRAM-bound kernel (≈ 69 misses per 1000
instructions, ~0 % LLC hits, IPC < 0.75) into a compute-bound one
(≈ 5 misses/1000 instructions, IPC ≥ 1.75). That single transformation —
not extra parallelism — is why the DAG is up to 9.3× faster on the same host,
and why its lead *shrinks* as plain DP throws 3× more total cycles at the
problem with 32 threads.

---

## 5. Findings in one paragraph

Plain KnapsackDP is memory-bound: ~70–100 LLC misses per 1000 instructions,
LLC hit ratio ≈ 0 %, IPC ≤ 0.75, and its 32-thread speedup (≈ 7.4 of 32) is
limited by a 3× inflation of total cycles and per-core rates at 68–85 % of
serial. The tiled DAG fixes the memory behaviour — LLC MPKI drops by an order
of magnitude (≈ 5), IPC reaches 2.4 — making it up to 9.3× faster at p = 1 on
the same host, but its thread scaling is nearly flat (≈ 1.0–1.2× from 1 to 32
threads): its win is locality and reduced scheduling overhead, not
parallelism. Within the swept block sizes, tile *count* — not cache misses —
explains the time differences between configurations: large `cap_block` wins
by executing 5–45× fewer DAG events, and too-large tiles (22 at cap 1e5) start
to starve the threads.

## 6. Caveats and known artifacts

- **Multiplexing:** all runs measure at 33–56 % coverage; ratios are
  trend-reliable, absolute values carry a few percent of noise. `llc_hit_pct`
  can go slightly negative (e.g. −1.65) when the two LLC event estimates
  disagree within noise at low coverage — treat |hit| < ~2 % as "≈ 0 %".
- **Cycles sum over all cores:** never read `cycles(p) < cycles(1)` as "fewer
  cycles per core"; use `rate`/`r` for the per-core view.
- **Timing skew:** `rate` divides perf cycles by *clean* wall time; the perf
  run and the clean run differ by scheduling noise (small relative to the
  ≥ 60 ms runs analysed).
- **Hosts:** plain-DP timings are from epito02, DAG timings from epito01;
  the cross-algorithm time comparison deliberately uses the epito01 plain-DP
  baselines embedded in `result_kpdpdag.json`.
- **Counter-grid mismatch:** the two perf sweeps cover different capacity
  ranges (plain DP: 1e6/1e7; DAG: 1e4/1e5/1e6), so the only directly comparable
  capacity is 1e6 — which is what §4.9 uses.
