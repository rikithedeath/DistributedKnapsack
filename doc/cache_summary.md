# Analisi dei contatori hardware: comportamento della cache in KnapsackDP e KnapsackDPDAG

> ⚠️ **Avvertenza sulle misure della DP classica.** Dal commit "complete
> knapsackdp granularization" (15/09/2026) fino alla revisione "Benchmark
> fairness: row-major classic DP" la DP classica usava una
> `Eigen::MatrixX<int>`, che Eigen memorizza **per colonne**: con il ciclo
> interno sulla capacità due celle consecutive distavano `(n+1)·4` byte.
> I tempi e i contatori della classica raccolti in quel periodo misurano quel
> layout, non l'algoritmo, e i confronti DAG/classica che li usano vanno
> rifatti (`sbatch/run_timing.sbatch`). Le misure precedenti al 15/09 usavano
> `vector<vector<int>>`, con righe contigue, e non ne sono toccate: controllare
> la data di ogni campagna. Il report aggiornato è `doc/main.typ`.

Questo documento riassume i risultati delle misure `perf stat` raccolte nei
file `doc/data/perf/perf_kpdp.csv`, `doc/data/perf/perf_kpdpdag.csv` e
`doc/data/perf/perf_kpdpbig.csv` e visualizzate nei grafici
`doc/images/perf_*.png`. È complementare a
[`benchmark_tiled_dp_dag.md`](benchmark_tiled_dp_dag.md) (confronto dei tempi)
e [`tiled_dp_dag.md`](tiled_dp_dag.md) (descrizione dell'algoritmo): qui
guardiamo **dentro** il processore per spiegare *perché* i tempi sono quelli.

- **Scopo**: quantificare il comportamento della gerarchia di memoria
  (L1, LLC, TLB), il tasso di istruzioni per ciclo (IPC) e il lavoro per
  cella di tabella, per separare i colli di bottiglia algoritmici da quelli
  di contesa/sincronizzazione.
- **Metodo**: ogni riga dei CSV è un evento `perf stat -x ,` di un run; gli
  11 eventi vengono ruotati in colonne e tutte le derivate ricalcolate dai
  contatori grezzi. **Tempi e contatori provengono dallo stesso run**: gli
  sbatch ripetono ogni configurazione 3 volte (`REPEATS=3`) e registrano
  nelle colonne finali `repeat` e `time_s` il tempo del solver letto sullo
  stdout di kpbench nella stessa esecuzione misurata da perf. Ogni
  configurazione è riassunta con la **mediana** delle 3 repliche (contatori
  e tempi); le repliche restano nei dati per le diagnostiche di
  ripetibilità. Niente join con i JSON della vecchia campagna: i tempi di
  `result_kpdp.json`/`result_kpdpdag.json` provengono da run e binari
  diversi e non vengono più usati in questa analisi.

---

## 1. Setup della misura

| | `perf_kpdp.csv` | `perf_kpdpdag.csv` |
|---|---|---|
| Solver | `knapsackdp` | `knapsackdpdag` |
| Configurazioni × repliche | 12 × 3 = 36 run | 360 × 3 = 1080 run |
| Griglia | max_weight {10, 1000} × capacity {1e6, 1e7} × threads {1, 8, 32} | max_weight {10, 100, 1000} × capacity {1e4, 1e5, 1e6} × threads {1,2,4,8,16,32} × item_block {5,10,15,25} × cap_block {1000,10000} |
| Processo misurato | `./build/kpbench` standalone (solo solver, nessun interprete Python) | idem |
| PMU | ARM `armv8_pmuv3` (core) + `ll_cache_rd`/`ll_cache_miss_rd` per l'ultimo livello | idem |
| Eventi | cycles, instructions, cache-references, cache-misses, branch-misses, L1-dcache-loads/misses, ll_cache_rd/miss_rd, dTLB/iTLB-load-misses | idem |
| Tabella DP | sempre `n = 50` item, quindi `(50+1) × (capacity+1)` celle | idem |

Note di affidabilità (riportate, mai usate per scartare dati):

- **ripetibilità dei tempi same-run**: spread massimo/minimo delle 3
  repliche per configurazione — KnapsackDP: mediana **1.011**, massimo
  **1.051** (0/12 config sopra 1.1); DAG: mediana **1.032**, massimo
  **1.385** (12/360 sopra 1.1, tutti run da ~1–2 ms). I tempi a muro della
  campagna sono quindi ripetibili al ~3%.
- **copertura PMU**: il contatore con la peggiore copertura di ogni run è
  riportato come `coverage_pct`. Per KnapsackDP è 51–54% (multiplexing
  intenso ma costante); per il DAG scende fino al **6%** sui run più corti
  (mediana 49%).
- **cross-check**: le metriche derivate di perf (`insn per cycle`, hit rate
  in %) sono state ricalcolate dai contatori grezzi: 108 + 3240 confronti
  (per replica), 0 disaccordi oltre l'arrotondamento al 2° decimale.
- **consistenza cicli/tempi**: il clock implicito `cycles / (T·p)` per
  KnapsackDP è **sempre in banda** (2.3–3.0 GHz, 0/12 fuori): tempi e
  contatori ora raccontano la stessa storia. Per il DAG 105/360
  configurazioni escono dalla banda (84 sopra 3.5 GHz, 21 sotto 1 GHz) ma
  sono quasi tutte run da **~2 ms** a p=1, dove l'estrapolazione dei
  contatori moltiplicati per la copertura non ha senso fisico (§3.3).

Notazione: `w` = famiglia di max_weight, `c` = capacity, `p` = thread;
`IPC` = istruzioni/ciclo, `MPKI` = miss per 1000 istruzioni,
`K_c` = `cycles / ((n+1)·(c+1))` = cicli per cella di tabella (somma su
tutti i core, quindi cresce anche con lo spinning).

---

## 2. KnapsackDP: il baseline è memory-bound e la parallelizzazione non lo cura

Tabella chiave (mediane della campagna perf, `perf_kpdp.csv`):

| w | c | p | T (s) | S_wall | IPC | K_c | LLC MPKI | dTLB MPKI | L1 hit | insn_ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 1e6 | 1 | 0.642 | 1.00 | 0.64 | 38.1 | 97.5 | 3.82 | 88.0% | 1.00 |
| 10 | 1e6 | 8 | 0.134 | 4.78 | 0.49 | 54.5 | 90.8 | 3.43 | 88.5% | 1.09 |
| 10 | 1e6 | 32 | 0.086 | 7.44 | 0.25 | 128.2 | 73.6 | 2.81 | 89.7% | 1.33 |
| 10 | 1e7 | 1 | 6.579 | 1.00 | 0.63 | 38.6 | 96.8 | 3.59 | 87.9% | 1.00 |
| 10 | 1e7 | 32 | 0.932 | 7.06 | 0.19 | 145.7 | 87.7 | 3.36 | 88.6% | 1.12 |
| 1000 | 1e6 | 1 | 0.677 | 1.00 | 0.61 | 40.1 | 102.2 | 4.81 | 82.5% | 1.00 |
| 1000 | 1e7 | 32 | 0.940 | 7.29 | 0.18 | 148.7 | 93.8 | 4.13 | 83.7% | 1.12 |

### 2.1. Ogni accesso alla LLC è un miss

L'hit ratio su `ll_cache` è **0–3%** in tutte le configurazioni
(`llc_hit_pct`): la tabella DP (40 MB a c=1e6, 400 MB a c=1e7) supera di
molto l'ultimo livello di cache e il layout **column-major** della tabella
(vedi l'avvertenza in testa) fa sì che ogni cache line, che contiene celle
di righe diverse, venga usata per un solo intero prima di essere espulsa. Il risultato è
**~90–100 LLC miss per 1000 istruzioni**: il sollevamento (lift) della
ricorrenza non ha località temporale sfruttabile. L'hit ratio L1 resta
alto (83–90%) perché ogni cella viene scritta e riletta a breve distanza
(la cella `dp[i-1][w]`, stessa colonna), ma il traffico verso la memoria è dominante:
con IPC ≈ 0.6 il core passa la maggior parte del tempo ad aspettare.

### 2.2. Il degrado per-cella è legato ai thread, non alla capacità

Grafico: [`perf_kpdp_tlb_memory_wall.png`](images/perf_kpdp_tlb_memory_wall.png)
e [`perf_kpdp_cycles_per_cell.png`](images/perf_kpdp_cycles_per_cell.png).

- `K_c` è **piatto in capacity** a p fisso (38→39 cicli/cella da 1e6 a 1e7)
  e cresce **col numero di thread** (38 → 146–149 a 32 thread).
- Il tasso di **dTLB miss resta ~3–5 MPKI** su tutto l'intervallo: anche se
  la tabella da 400 MB eccede di gran lunga la copertura della TLB, il
  page-walk non è ciò che degrada le prestazioni.
- Verdetto: il degrado di `K_c` è **contesa/sincronizzazione** (spinning
  sulle barriere OpenMP, condivisione delle cache line di confine tra le
  porzioni di riga), non TLB né capacità. Lo conferma `insn_ratio` > 1:
  a 32 thread vengono eseguite fino al **33% di istruzioni in più** della
  versione seriale (spinning visibile ai contatori).

### 2.3. IPC in collasso e scomposizione dello speedup

Grafici: [`perf_kpdp_ipc_vs_threads.png`](images/perf_kpdp_ipc_vs_threads.png),
[`perf_kpdp_memory_intensity.png`](images/perf_kpdp_memory_intensity.png),
[`perf_kpdp_decomposition.png`](images/perf_kpdp_decomposition.png).

- L'IPC aggregato crolla da 0.63 (p=1) a 0.18–0.25 (p=32): i cicli extra
  sono attesa, non lavoro.
- Scomposizione `S_wall = S_cyc · p · r` a 32 thread: S_wall ≈ 7.1–7.7 si
  compone di `S_cyc ≈ 0.27–0.31` (il numero totale di cicli **cresce** di
  ~3–3.8× per via dell'overhead) moltiplicato per `p·r ≈ 23–25` (il
  recupero dato dai core, con efficienza per-core `r ≈ 0.77–0.84`).
- I branch miss sono trascurabili (0.01–0.02% delle istruzioni): il collo
  di bottiglia è interamente la memoria/sincronizzazione.
- Curiosità controintuitiva: LLC MPKI **diminuisce** con i thread
  (98 → 74 a w=10, c=1e6). Con più core che elaborano porzioni di riga in
  parallelo, la stessa cache line confinaria viene riutilizzata da più
  core prima di essere espulsa: meno miss pro capite, ma più contesa.

---

## 3. KnapsackDPDAG: il tiling risolve il problema della cache

### 3.1. L'effetto tiling: 18–28× miss in meno, IPC ×3.5–4.3, ~4× sul tempo seriale

Grafici: [`perf_vs_algorithms.png`](images/perf_vs_algorithms.png),
[`perf_kpdpdag_block_grid.png`](images/perf_kpdpdag_block_grid.png),
[`perf_kpdpdag_miss_vs_tiles.png`](images/perf_kpdpdag_miss_vs_tiles.png).

Confronto diretto a capacity = 1e6: **entrambi gli algoritmi provengono
dalla stessa campagna perf** (stesso host, stesso binario, tempi same-run),
quindi tempi, IPC, K_c e MPKI sono direttamente confrontabili:

| w | p | T_dag/T_kpdp | config | IPC kpdp → dag | K_c kpdp → dag | miss LLC kpdp → dag | riduzione |
|---|---|---|---|---|---|---|---|
| 10 | 1 | **0.252** | ib25/cb1000 | 0.64 → 2.29 | 38.1 → 10.8 | 121.0M → 5.0M | **24×** |
| 10 | 8 | 0.495 | ib10/cb1000 | 0.49 → 2.36 | 54.5 → 21.6 | 123.0M → 6.8M | 18× |
| 10 | 32 | 0.782 | ib5/cb10000 | 0.25 → 2.61 | 128.2 → 47.0 | 121.3M → 6.4M | 19× |
| 1000 | 1 | 0.261 | ib25/cb10000 | 0.61 → 2.16 | 40.1 → 10.8 | 127.6M → 4.6M | 28× |
| 1000 | 8 | 0.536 | ib10/cb10000 | 0.49 → 2.24 | 55.9 → 18.1 | 128.7M → 5.2M | 25× |
| 1000 | 32 | 0.788 | ib5/cb10000 | 0.25 → 2.63 | 129.2 → 47.9 | 134.2M → 6.6M | 20× |

Il tiling tiene il working set di ogni tile nella cache: in seriale a
parità di istruzioni eseguite (~1.2–1.25 miliardi per entrambi), i miss
LLC crollano di un fattore **18–28×**, l'IPC sale a **2.2–2.3** e il
numero di cicli per cella scende da ~38–40 a ~11. A p=1 il DAG è **~4×
più veloce** del plain DP (T_dag/T_kpdp ≈ 0.25): il guadagno è quasi tutto
località, non parallelismo. (Nella prima stesura di questo documento, basata
sui tempi della vecchia campagna JSON, si leggeva "~9×": era un artefatto
del confronto tra campagne diverse, come aveva fatto sospettare il check
del clock implicito. Con i tempi same-run il rapporto corretto è ~4×.)

### 3.2. Effetto della granularità dei tile

- **A p basso vince la granularità grossa**: `ib25` è la configurazione
  ottima a p=1–2 su tutte le capacity, `ib15` a p=4, `ib10` a p=8: più
  thread richiedono tile più piccoli per avere abbastanza lavoro
  indipendente. La mappa delle configurazioni vincenti è in
  [`perf_kpdpdag_best_config_map.png`](images/perf_kpdpdag_best_config_map.png).
- **Il ruolo di cap_block è secondario**: a parità di (w, c, p) le due
  scelte di cap_block differiscono di pochi punti percentuali (a c=1e6,
  p=1, w=10: con ib25 la mediana è 0.162 s con cb1000 e 0.167 s con
  cb10000; con ib5 l'ordine si inverte, 0.215 s contro 0.210 s). Il vecchio
  ASSERT "cb10000 batte quasi sempre", derivato dalla campagna JSON, non si
  conferma: `cb10000` diventa preferibile solo a c=1e6 (per w=1000 a tutte
  le p, per w=10/100 da p=16) e sporcamente altrove.
- A capacity = 1e4 (tabella di 51×10001 = 510k celle, ~2 MB: entra in
  cache) domina `ib25/cb1000` a p basso; il numero di tile resta piccolo
  (22–55) e la località non è il problema.

### 3.3. Affidabilità dei tempi del sweep DAG

Grafico: [`perf_kpdpdag_reliability.png`](images/perf_kpdpdag_reliability.png).

Con i tempi same-run il quadro cambia rispetto alla vecchia analisi:

- **288 run su 360 durano meno di 100 ms** (i più corti ~1–2 ms): la
  copertura PMU scende fino al 6% (mediana 34% sotto i 50 ms). I *tempi*
  però sono ripetibili (spread mediano 1.032); sono i **contatori
  estrapolati** a essere inaffidabili sui run più corti: lì il check del
  clock implicito produce valori assurdi (fino a 17 GHz "impliciti" su
  run da 2 ms con copertura 15–50%). Le 12 configurazioni con spread >1.1
  sono concentrate nei run da 1–9 ms (quasi tutte a c=1e4).
- **Lo scaling ora è visibile dove il calcolo lo consente**: a c=1e6 la
  miglior configurazione passa da 162 ms (p=1) a 65–68 ms (p=16–32),
  `S_wall` fino a **2.5–2.6×**; a c=1e5 da 16.5 ms a 7.0–8.0 ms (fino a
  2.4×). A c=1e4 (run da 1–2 ms) il tempo è piatto e a p=32 **peggiora**
  (S_wall 0.73–0.83): l'overhead di sincronizzazione domina il calcolo.
- I contatori raccontano il costo di questo parallelismo: `S_cyc` scende
  regolarmente 0.92 → 0.22, ma `insn_ratio` esplode fino a **4.6×**
  (c=1e6) e **~9×** (c=1e4) a 32 thread — a tabella piccola quasi tutte le
  istruzioni extra sono spinning sulle barriere OpenMP — e `rate_ratio`
  crolla a 0.10–0.35.
- Regola pratica aggiornata: per confrontare configurazioni del DAG i
  tempi same-run (mediana di 3 repliche) sono affidabili; i contatori
  scalati (MPKI, insn_ratio) vanno letti solo sui run ≥ ~50 ms, dove la
  copertura supera il 45%.

---

## 4. Sweep n grande: crossover dello scaling e banda DRAM

Il limite di scaling del DAG osservato in §3 (S_wall ≈ 2.5× a c=1e6, n=50) è
stato messo alla prova con lo sweep `run_kpdpdag_big.sbatch`: stessa capacity
(1e6, tabella sempre ben oltre la cache) ma `num_items` portato da 50 a
**1000 e 5000**, cioè 20×/100× il lavoro per tile e tra sincronizzazioni.
Configurazione: DAG item_block {10,25} × cap_block {1000,10000}, max_weight=10,
p {1,4,8,16,32}, 2 repliche, più una baseline plain DP (w=10) nella stessa
campagna. Qualità della misura: ripetibilità eccellente (spread max/min
≤ **1.03** su tutte le 50 configurazioni), copertura 53–54%, clock implicito
in banda (DAG 0.77–3.33 GHz, plain 2.40–2.99 GHz).

Grafico: [`perf_big_crossover.png`](images/perf_big_crossover.png).

### 4.1. Lo scaling del DAG cresce col lavoro per tile: 2.5× → 7.1× → 9.1×

| n | T_dag(p=1) | T_plain(p=1) | vantaggio seriale DAG | S_wall DAG p=32 | S_wall plain p=32 | insn_ratio DAG p=32 |
|---|---|---|---|---|---|---|
| 50 (§3) | 0.16 s | 0.64 s | 4.0× | 2.4× | 7.4× | 4.4× |
| 1000 | 3.41 s | 23.10 s | 6.8× | 7.1× | 8.8× | 1.35× |
| 5000 | 15.9 s | 150 s | 9.4× | 9.1× | 14.8× | 1.25× |

- **L'ipotesi sync-bound è confermata**: a 20–100× più lavoro per tile lo
  scaling a muro del DAG sale da 2.4× a **7.1×** (n=1000) e **9.1×**
  (n=5000), e `insn_ratio` crolla da 4.4× a ~1.25×: lo spinning sulle
  code/atomiche è ora una frazione marginale del lavoro (`s_cyc` DAG a
  p=32: 0.45–0.62, contro 0.23 a n=50).
- **Il vantaggio seriale del DAG cresce con n**: 4× → 6.8× → 9.4×. La
  ragione è nel costo per cella: il plain DP seriale passa da 12.6
  (n=50) a 23.1 (n=1000) e 30.0 ns/cella (n=5000) — la tabella da 4–20 GB
  peggiora cache e TLB — mentre il DAG resta **piatto a ~3.2–3.4 ns/cella**
  per ogni n: il tiling protegge la località anche al crescere della
  tabella (a n=5000, p=32: 0.10 miss LLC e 0.044 miss dTLB per cella).
- **A n grande il DAG domina a tutte le p**: a p=32 e n=5000 fa 1.75 s
  contro 10.1 s del plain (**5.8× più veloce**), e `t_dag/t_plain` scende
  da 0.25 (n=50) a 0.17.
- Il plain, a sua volta, scala meglio con n grande: S_wall 14.8× a p=32
  (n=5000, ~46% di efficienza) contro 7.4× a n=50 — anche le sue barriere
  per-riga si ammettono quando ogni riga costa di più.

### 4.2. Banda DRAM stimata: non è (ancora) il muro

Grafico: [`perf_bandwidth.png`](images/perf_bandwidth.png).

Stima del traffico DRAM (`add_bandwidth` in
[`kpdp_perf_analysis.py`](kpdp_perf_analysis.py)): letture =
`ll_cache_miss_rd` × 64 B (limite inferiore: i prefetch che colpiscono la
cache sfuggono al contatore), scritture = 4 B per cella di tabella (ogni
cella è scritta una volta); divise per il tempo same-run.

- **Plain DP**: da ~12 GB/s in seriale a **92–100 GB/s** a p=32 (n=50,
  c=1e6) e **121 GB/s** (n=1000, p=32). Le curve crescono monotonamente
  con p e **non mostrano plateau**: a 32 thread la banda DRAM non è ancora
  il limite del plain — il limite è la sincronizzazione (`r ≈ 0.77–0.83`,
  `insn_ratio` > 1 per spinning).
- **DAG**: 3 GB/s in seriale → 22–30 GB/s a p=32: operazioni con un
  ordine di grandezza meno traffico.
- **Traffico totale** a n=5000, p=32: DAG **33 GB** letti contro **569 GB**
  del plain — **17× meno traffico DRAM** per lo stesso problema. Questo è
  il costo reale del "memory-bound by construction" del plain DP, e la
  risorsa che il tiling restituisce.

## 5. Verdetto

1. **Il plain DP è memory-bound by construction**: ~90–100 LLC MPKI, hit
   L3 ~0–3%, IPC ~0.6. Più thread non aggiungono località: aggiungono
   contesa (`K_c` ×4, IPC ÷3, fino a +33% di istruzioni di spinning).
2. **Il tiling del DAG attacca la causa giusta**: stesso numero di celle,
   stesso numero di istruzioni in seriale, ma **18–28× meno miss LLC** e
   IPC ×3.5–4.3 → ~11 cicli/cella contro ~38–40, e **~4× sul tempo a muro
   seriale** a c=1e6 con n=50 (rapporto corretto dopo l'adozione dei tempi
   same-run; la vecchia stima "~9×" era un artefatto cross-campagna).
3. **Il limite di scaling di entrambi è la sincronizzazione, non la banda
   DRAM** — e dipende dal rapporto calcolo/sincronizzazione, come verifica
   lo sweep n grande (§4): con 20–100× più lavoro per tile lo scaling del
   DAG sale da ~2.4× a **7.1×/9.1×** a p=32 e il suo spinning (`insn_ratio`)
   crolla da 4.4× a ~1.25×; il plain, a sua volta, sale da 7.4× a 14.8×.
   Le stime di banda restano monotone crescenti (92–121 GB/s a p=32,
   senza plateau): il collo è sync/latenza, che il lavoro maggiore ammette.
4. **Il tiling protegge la località al crescere di n**: il costo per cella
   del DAG è piatto (~3.2–3.4 ns/cella) mentre quello del plain degrada
   (12.6 → 30.0 ns/cella da n=50 a n=5000): a n grande il DAG è più
   veloce del plain a *tutte* le p (5.8× a p=32, n=5000) e muove **17×
   meno traffico DRAM**.
5. **Cosa usare a questa scala**: tempi same-run ripetuti (spread ≤ ~3%)
   per confrontare configurazioni; IPC e MPKI dei run ≥ ~50 ms per
   spiegare *perché*. I contatori scalati dei run da pochi ms e il check
   del clock implicito restano diagnostiche di affidabilità, non dati.

## 6. Riproduzione

```bash
# misure (nodo ARM con permessi perf; REPEATS=3, tempi same-run nelle
# colonne finali repeat,time_s di perf_summary.csv):
sbatch sbatch/run_kpdp_perf.sbatch      # -> perf_kpdp_out/perf_summary.csv
sbatch sbatch/run_kpdpdag_perf.sbatch   # -> perf_kpdpdag_out/perf_summary.csv
sbatch sbatch/run_kpdpdag_big.sbatch    # -> perf_big_out/perf_summary.csv
# fetch (con backup dei CSV precedenti):
.opencode/skills/hpc-mercurio/scripts/fetch-results.sh

# analisi (tabelle a stdout + grafici in doc/images/):
~/pyenv/sci/bin/python doc/kpdp_perf_analysis.py
```

Dettagli delle formule (IPC, MPKI, K_c, `S_wall = S_cyc · p · r`,
`insn_ratio`, check del clock, aggregazione a mediana sulle repliche) nel
docstring di [`kpdp_perf_analysis.py`](kpdp_perf_analysis.py).
