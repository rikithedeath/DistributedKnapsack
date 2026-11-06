# Benchmark del Solver Tiled DP-over-DAG (Knapsack 0/1)

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

Questo documento è un **report sperimentale** complementare a
[`doc/tiled_dp_dag.md`](tiled_dp_dag.md), che descrive l'algoritmo in dettaglio.
Qui misuriamo e confrontiamo le prestazioni dei tre solver che risolvono lo
stesso problema dello zaino 0/1 su un **singolo nodo shared-memory**, sulla
macchina di riferimento (8 thread OpenMP).

- **Scopo**: quantificare il costo dei tre approcci, isolare i colli di
  bottiglia (barrier, parallelismo, per-cella) e trarre un verdetto onesto su
  dove l'approccio a DAG abbia (o non abbia) senso.
- **Invariante di correttezza**: in ogni scenario i tre solver producono
  **valori e pesi bit-identici** (stessi Profit e Weight totali).
- **Regime misurato**: questo report copre lo scenario "molti item,
  capacity piccola/media" su un nodo a 8 thread. La campagna `perf`
  documentata in [`cache_summary.md`](cache_summary.md) — n=50, capacity
  fino a 1e6–1e7, fino a 32 thread, tempi same-run ripetuti 3 volte con
  contatori hardware — copre il regime complementare "pochi item, capacity
  grande": le conclusioni dei due documenti non si contraddicono, perché i
  regimi sono opposti (§5.4).

---

## 1. I tre solver sotto test

Tutti risolvono il knapsack 0/1 con DP; differiscono in **come** organizzano e
parallelizzano il riempimento della tabella `dp[n+1][capacity+1]`.

### 1.1. `knapsackdp` (baseline classica)

File: `src/Knapsack/knapsackdp.cpp`.

Loop classico riga-per-riga con parallelismo OpenMP **orizzontale sulla capacità**:

```cpp
for (int i = 1; i <= n; ++i) {
    #pragma omp parallel for
    for (int w = 0; w <= capacity; ++w) {
        dp[i][w] = ...;   // skip / take
    }
}
```

È il **riferimento di baseline**: una riga alla volta, con un solo passo di
dipendenza `dp[i-1][·]` letto a monte. La ricorrenza legge la riga
precedente nelle colonne `w` e `w - w_i`, quindi con la tabella
**memorizzata per righe** la visita è sequenziale e amica del prefetcher.

### 1.2. `knapsackdpdag` / `solve_dag` (levelized)

File: `src/Knapsack/knapsackdpdag.cpp`.

Partiziona la tabella in **tile** rettangolari (`item_block` righe × `cap_block`
colonne), costruisce un DAG `boost::adjacency_list` delle dipendenze tra tile (via
`build_graph`), calcola i livelli con `compute_levels`, quindi esegue il calcolo
**per wavefront**:

```cpp
for (int L = 0; L <= max_level; ++L) {
    START_BLOCK("KnapsackDPDAG::LevelCompute");
    #pragma omp parallel for
    for (std::size_t t = 0; t < tiles.size(); ++t)
        compute_tile(g, tiles[t], weights, values, item_block, cap_block);
    END_BLOCK("KnapsackDPDAG::LevelCompute");
}
```

C'è **una barriera globale per livello** (`#pragma omp parallel for` termina a
ogni `L`). Con `levels` dell'ordine delle centinaia–migliaia, questo significa
altrettante barriere. Il codice completo della fase di calcolo è in `solve_dag`.

### 1.3. `solve_dag_topo` (schedulatore topologico, NOVO)

Stesso file, schedulatore **counter-based / work-queue**:

- un `std::vector<std::atomic<int>> indeg` ricavato da `boost::in_degree`;
- un **contatore atomico rimanente** `remaining` (tile ancora da calcolare);
- una **ready-stack lock-free a doppio contatore**: i produttori riservano uno
  slot con `tail.fetch_add`, scrivono il valore, poi lo pubblicano con una
  `release` store su `filled`; i consumatori leggono `head < filled` (acquire)
  e fanno `compare_exchange` su `head`. Ordine release/acquire garantisce la
  visibilità del tile pronto.
- un **team di worker `#pragma omp parallel`**: ogni worker prende un tile
  pronto, esegue `compute_tile`, poi decrementa i successori con
  `fetch_sub` e li spinge in coda quando arrivano a zero.

Nessuna barriera per livello: il parallelismo emerge direttamente dal conteggio
delle dipendenze. Selezionabile tramite la variabile d'ambiente
`KNAPSACK_DAG_TOPO` (default = `1`/topo; `"0"` = levelized).

### 1.4. Kernel condiviso `compute_tile`

Entrambi i solver DAG chiamano **la stessa funzione `compute_tile`**: copia della
riga boundary + ciclo interno `for a / for c` con la ricorrenza DP skip/take.
Di conseguenza $solve\_dag$, $solve\_dag\_topo$ e $knapsackdpdag$ producono
**risultati bit-for-bit identici**.

L'end-point pubblico `knapsackdpdag()` (in `src/Knapsack/knapsackdpdag.cpp`)
orchestra: `build_graph` → `solve_dag_topo` (default) → `reconstruct_items`.

---

## 2. Strumentazione e metriche

### 2.1. `DAGStats`

Le statistiche della struttura DAG sono raccolte **serialmente** *fuori* da ogni
regione parallela — necessariamente, perché il timer `START_BLOCK`/`END_BLOCK`
(in `src/time.hpp`) serializza su un accumulatore di sezione a singolo owner.
`knapsackdpdag()` le popola e le espone:

```cpp
g_last_dag_stats.tiles = nb * nq;                  // numero di tile (vertici)
g_last_dag_stats.edges = (int)boost::num_edges(g); // numero di archi
g_last_dag_stats.levels = max_level + 1;           // occupazione dei livelli
g_last_dag_stats.maxFrontier = max_frontier;       // max|F_L|, wavefront più ampio
```

dove `DAGStats{tiles, edges, levels, maxFrontier}` è esposto a Python via pybind
(`m.def("get_dag_stats", &get_dag_stats)` in `src/lib.cpp`) e stampato da
`distributedknapsack/testengine.py`:

```
DAG stats: tiles=... edges=... levels=... maxFrontier=...
```

### 2.2. Tempi raccolti

- **`knapsackdp`**: tempo totale del solo calcolo (sezione `KnapsackDP::Compute`).
- **`topo` / `levelized`**: tempo totale di `knapsackdpdag` (GraphBuild + Dag Solve
  + stats seriali). Quando riportati, i tempi `split` separano le sezioni
  `GraphBuild` e `Dag Solve`.

---

## 3. Miglioramenti di codice introdotti

Oltre al nuovo schedulatore topologico, il branch ha incluso alcune correzioni e
rifiniture rilevanti per la correttezza e per la prestazione:

1. **`compute_tile` condiviso**: entrambi i solver DAG invocano lo stesso
   kernel di tile, per garantire risultati bit-for-bit identici (basta un unico
   percorso di codice da validare).
2. **Edge dedup in `build_graph`**: gli intervalli di dipendenza per item sono
   raccolti, ordinati e **fusi** (`merged`/`merged2`) prima di creare gli archi;
   ogni arco sorgente viene aggiunto **una sola volta**, senza alterare l'insieme
   degli archi né i livelli (controllo di monotonia `lo_i <= merged2.back().second + 1`
   per unire intervalli adiacenti).
3. **Niente zeroing dei blocchi**: le celle di ogni `BlockMatrix` vengono
   sovrascritte da `solve_dag`/`solve_dag_topo` prima di essere lette (riga 0 per
   tutte le `c`, poi le righe `a+1`), quindi si evita di azzerare la matrice alla
   creazione dei vertici.
4. **Fix del bug `compute_levels`**: in precedenza la chiamata
   `compute_levels(g);` era stata "inghiottita" da un commento (`//`) e i
   wavefront levelized si calcolavano su livelli non inizializzati (es. tutti a
   livello 0 ⇒ nessun parallelismo). La chiamata è ora eseguita prima della fase
   `by_level` in `solve_dag`.

---

## 4. Risultati

**Condizioni comuni**: `OMP_NUM_THREADS=8`, `seed=42`, `maxWeight` come indicato
in ogni scenario. Profit e Weight sono **identici** tra i tre solver in ogni
scenario (invariante di correttezza). Tempi in secondi.

### 4.1. Scenario (a)

```
numItems=300000, capacity=128, maxWeight=64, item_block=1500, cap_block=16
Profit 6367 / Weight 128
```

| Solver        | Tempo (s) | Note                                    |
|---------------|-----------|------------------------------------------|
| `knapsackdp`  | 1.09      | baseline                                 |
| topo          | 1.42      | split: GraphBuild 688ms, Dag Solve 768ms |
| levelized     | 1.48      |                                          |

DAG: `tiles=1800, edges=13956, levels=208, maxFrontier=9`.

### 4.2. Scenario (b)

```
numItems=100000, capacity=500, maxWeight=100, item_block=500, cap_block=50
Profit 19484 / Weight 500
```

| Solver        | Tempo (s) | Note                                    |
|---------------|-----------|------------------------------------------|
| `knapsackdp`  | 0.63      | baseline                                 |
| topo          | 1.18      | split: GraphBuild 295ms, Dag Solve 905ms |
| levelized     | 1.77      |                                          |

DAG: `tiles=2200, edges=11959, levels=210, maxFrontier=11`.

**Nota**: in questo scenario la topo batte il levelized di ≈ **−33%**
(1.18 s vs 1.77 s).

### 4.3. Scenario (c)

```
numItems=20000, capacity=20000, maxWeight=100, item_block=200, cap_block=2000
Profit 78381 / Weight 20000
```

| Solver        | Tempo (s) |
|---------------|-----------|
| `knapsackdp`  | 2.56      |
| topo          | 6.63      |
| levelized     | 10.3      |

DAG: `tiles=1100, edges=4168, levels=110, maxFrontier=11`.

### 4.4. Sweep sulle dimensioni dei tile (solo topo)

```
numItems=100000, capacity=500, maxWeight=100   (Profit/Weight = 19484/500 sempre)
```

| item_block | cap_block | Tempo (s) | tiles   | edges    | levels | maxFrontier |
|------------|-----------|-----------|---------|----------|--------|-------------|
| 16         | 32        | 2.09      | 100000  | 801312   | 6265   | 16          |
| 16         | 128       | 2.02      | —       | —        | —      | 4           |
| 16         | 250       | 3.47      | —       | —        | —      | 3           |
| 64         | 32        | 1.48      | —       | —        | —      | 16          |
| 64         | 128       | 1.83      | —       | —        | —      | 4           |
| 64         | 250       | 3.29      | —       | —        | —      | 3           |
| **256**    | **32**    | **1.31**  | —       | —        | —      | **16**      |
| 256        | 128       | 1.79      | —       | —        | —      | 4           |
| 256        | 250       | 3.27      | —       | —        | —      | 3           |

Configurazione **migliore dello sweep**: `item_block=256, cap_block=32`,
1.31 s, split `GraphBuild 422ms`, `Dag Solve 863ms`, `maxFrontier=16`.

---

## 5. Analisi

### 5.1. Il schedulatore topologico batte il levelized

In ogni scenario il `solve_dag_topo` è più veloce del `solve_dag` levelized,
con un vantaggio marcato dove la differenza è misurata:

- scenario (b): 1.18 s vs 1.77 s (≈ −33%);
- scenario (c): 6.63 s vs 10.3 s.

Questo conferma che le **barriere globali per-livello** del levelized erano un
costo reale: con `levels` tra ~110 e 6265, il livello basato su wavefront
equivale a **centinaia–migliaia di barriere** `#pragma omp parallel for`, ognuna
delle quali serializza il pipe. Il work-queue topologico elimina le barriere e
lascia lavorare i worker in modo asimmetrico, otto alla volta.

### 5.2. ... ma in questo regime il DAG non batte mai il DP classico

Nonostante questi guadagni interni, **nessuna configurazione di `knapsackdpdag`
(sia topo sia levelized, in nessuna dimensione di tile) supera il classico
`knapsackdp`** in tempi di pareggio:

| Scenario | `knapsackdp` | migliore topo | gap |
|----------|--------------|---------------|-----|
| (a)      | 1.09         | 1.42          | +30%|
| (b)      | 0.63         | 1.18          | +87%|
| (c)      | 2.56         | 6.63          | +159%|
| sweep (best) | —        | 1.31          | ≥ +108% vs 0.63 |

La ragione non è la mancanza di parallelismo: `maxFrontier` è 9–16, cioè **≥ 8
thread**, quindi il DAG espone tante tile indipendenti quanti sono i core. Il
vero punto cieco è che anche il **solo `Dag Solve`** (topo + compute, astraendo
dalla costruzione del grafo) costa già 768–905 ms, cioè **più dell'intero tempo
del `knapsackdp`** (0.63–1.09 s). Il collo di bottiglia, quindi, non è nelle
barriere né nel parallelismo.

(Questa conclusione vale per il regime "molti item, righe corte" di questi
scenari; nel regime complementare a tabelle grandi la campagna perf misura
l'esito opposto — §5.4.)

### 5.3. Causa radice: il costo per-cell del tiling

Il sospetto dominante è il **costo per-cella del tiling**. In scenari dove
`maxWeight` è grande rispetto a `cap_block`, la maggior parte delle letture
"take-item" a `left.block(a, c_prev)` cade in un **tile vicino** (scarsa locality
cross-tile): ogni tile paga

1. la **copia della riga boundary** (una lettura indiretta nel tile sopra);
2. **l'indirezione cross-tile** per ogni read che esce dal tile corrente.

Il DP classico riga per riga, invece, legge la riga precedente nelle colonne
`w` e `w - w_i`: se la tabella è memorizzata per righe (come nella versione
con `vector<vector<int>>` e in quella corretta) la visita è sequenziale e
amica del prefetcher; con la `Eigen::MatrixX<int>` column-major usata fra il
15/09 e la revisione non lo era (vedi l'avvertenza in testa). La partizione in tile — pensata per il futuro parallelismo MPI —
aggiunge overhead senza recuperarlo localmente.

Si aggiunge che la sola **`GraphBuild` costa già 295–688 ms** in queste
configurazioni: un onere fisso maggiore o uguale all'intero tempo del `knapsackdp`
di riferimento.

### 5.4. Relazione con la campagna perf: i due regimi non si contraddicono

La campagna di misure hardware di
[`cache_summary.md`](cache_summary.md) (`perf stat` su `kpbench`, nodo ARM
epito01, 3 repliche con tempi same-run, fino a 32 thread) copre il regime
opposto a quello di questo report:

| | Questo report | Campagna perf (`cache_summary.md`) |
|---|---|---|
| numItems | 20 000–300 000 | 50 |
| capacity | 128–20 000 | 10⁴–10⁷ |
| thread | 8 | 1–32 |
| esito DAG vs DP classico | sempre più lento (+30…+159%) | **~4× più veloce** a p=1 (c=1e6); con n grande il vantaggio cresce a 6.8–9.4× e il DAG vince a tutte le p (§4 di `cache_summary.md`) |

(I tempi assoluti dei due dataset non sono confrontabili: macchine e
banchi di prova diversi; il confronto è qualitativo.)

La sintesi: **l'esito dipende dal rapporto tra dimensione della tabella e
costo del tiling**.

1. **Capacity piccola (righe corte)** — regime di questo report. La
   scansione row-major del DP classico è sequenziale e prefetch-friendly;
   il DAG invece paga decine di migliaia di tile con `GraphBuild` da
   centinaia di ms, copia della riga boundary e indirezione cross-tile per
   ogni lettura che esce dal tile. Nessun guadagno di località da
   recuperare: il tiling è puro overhead.
2. **Capacity grande e n piccolo** — regime della campagna perf. La
   tabella (40–400 MB) eccede l'ultimo livello di cache: il DP classico
   fa ~90–100 LLC miss per 1000 istruzioni (hit L3 ~0–3%, IPC ~0.6)
   perché la riga precedente viene espulsa prima di essere rileta. Il
   tiling riporta il working set di ogni tile in cache: **18–28× meno
   miss LLC**, IPC 2.2–2.3, ~11 cicli/cella contro ~38–40, e **~4× sul
   tempo a muro seriale** a c=1e6. Con n=50 inoltre `GraphBuild` è
   trascurabile (a ib25/cb1000 la griglia ha ~2000 tile, non centinaia di
   migliaia).
3. **Il parallelismo, in entrambi i regimi, segue lo stesso criterio**: a
   c=1e6 con n=50 il DAG-best scala fino a ~2.5× a muro (p=16) prima che
   spinning e barriere (`insn_ratio` fino a ~4.6×) riassorbano il guadagno;
   l'esperimento di crossover di `cache_summary.md` (§4) conferma che il
   limite è il rapporto calcolo/sincronizzazione: con 20–100× più lavoro
   per tile lo scaling sale a 7.1×/9.1× a p=32 e lo spinning crolla a
   ~1.25×. A c=1e4 (run da 1–2 ms) l'overhead domina e a 32 thread il
   tempo peggiora — coerente con l'osservazione di questo report che il
   costo di sincronizzazione può annullare il parallelismo esposto.

In altri termini: il verdetto negativo di questo report vale **per tabelle
con righe corte**, dove il DP classico è già cache-ottimo; la vittoria del
tiled DAG nella campagna perf vale **per tabelle che eccedono la cache**,
dove il DP classico è memory-bound by construction.

---

## 6. Conclusioni (verdetto onesto)

1. **Il schedulatore topologico è un miglioramento reale di correttezza e
   prestazione** rispetto al levelized: elimina le barriere per-livello,
   avvia i wavefront senza sincronizzazioni globali e produce valori
   bit-identici. È da considerare la via d'accesso di default.
2. **Ma nel regime di questo report (molti item, righe corte) su un nodo
   shared-memory, la DP tiled su DAG non riesce a battere la classica DP
   row-parallel** `knapsackdp`, per qualsiasi dimensione di tile. Il costo
   per-cell del tiling (boundary copy + indirezione cross-tile, più la
   costruzione pesante del grafo) supera ogni guadagno di parallelismo,
   che pure esiste (`maxFrontier` ≥ 8). Nel regime complementare — n
   piccolo, capacity grande, tabella più grande della cache — la campagna
   perf mostra l'esito opposto: il tiled DAG batte il DP classico di ~4×
   in seriale (fino a 9.4× con n=5000) e, con abbastanza lavoro per tile,
   scala fino a ~9× a muro (§5.4 e §4 di `cache_summary.md`).
3. **Il vero valore del DAG è duplice**: nel regime "tabella >> cache" la
   località del tiling è un guadagno reale anche su un singolo nodo
   (campagna perf, §5.4); nel contesto MPI/distribuito la struttura a
   dipendenze rimuove inoltre la *cross-process data sharing* che né
   `omp for` né il DP classico possono esprimere. Nel regime "righe
   corte", invece, la baseline `knapsackdp` resta il punto di riferimento
   per il tempo.

**In sintesi**: per questo specifico problema, nel regime "molti item e
righe corte" la fattorizzazione `knapsackdpdag` = costruzione del grafo +
schedulatore topologico non è competitiva in locale — ma il codice di
schedulazione topologico e la 'clean-room' di `compute_tile`/edge-dedup
sono pronti e preziosi come base per l'esecuzione distribuita e per il
regime a tabelle grandi, dove la stessa macchina misura il DAG ~4× più
veloce del DP classico in seriale.

---

## 7. Riferimenti

- Sibling algoritmico (struttura DAG, tiling, wavefront): [`doc/tiled_dp_dag.md`](tiled_dp_dag.md)
- Campagna `perf` con contatori hardware e tempi same-run (regime n=50,
  capacity grandi): [`doc/cache_summary.md`](cache_summary.md)
- Sorgenti:
  - `src/Knapsack/knapsackdp.cpp` — DP classico row-parallel (baseline)
  - `src/Knapsack/knapsackdpdag.cpp` — `build_graph`, `compute_levels`,
    `compute_tile`, `solve_dag` (levelized), `solve_dag_topo` (topo),
    `knapsackdpdag`, `get_dag_stats`
  - `src/Knapsack/knapsackdpdag_impl.hpp` — `NodeData`, `Graph`, `DAGStats`,
    dichiarazioni
  - `src/lib.cpp` — binding pybind di `get_dag_stats` / `DAGStats`
  - `src/time.hpp` — `START_BLOCK`/`END_BLOCK` (timer serializzato a singolo owner)
  - `distributedknapsack/testengine.py` — stampa di `DAG stats` e dei tempi