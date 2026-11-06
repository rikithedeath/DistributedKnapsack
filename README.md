# DistributedKnapsack

Soluzioni parallele, in memoria condivisa, dello zaino 0/1 per programmazione
dinamica, con un benchmark C++ (`kpbench`), un modulo Python (pybind11) e il
report in Typst (`doc/main.typ`). Le varianti MPI, CUDA e COPA sono nel
repository ma fuori dal perimetro del report principale.

## Le varianti

Tutte condividono lo stesso kernel interno (`src/Knapsack/dp_kernel.hpp`), così
una differenza di tempo viene solo da visita, layout o sincronizzazione.

| `--test` | Cosa fa | Memoria |
|---|---|---|
| `knapsackdp` | classica: tabella completa row-major, ogni riga divisa fra i thread | 4 B per cella |
| `knapsackdpbits` | classica con due righe e un bit per cella per la ricostruzione | 1/8 B per cella |
| `knapsackdptiled` | tile sulla tabella completa, eseguiti per anti-diagonali, senza grafo | 4 B per cella |
| `knapsackdpdag` | tile con un DAG Boost.Graph delle dipendenze, eseguito per livelli | 4 B per cella + grafo |
| `bandwidth` | triade in stile STREAM su 3 vettori di `--capacity` double: la banda per il roofline | — |

## Compilare e provare

Dipendenze: CMake ≥ 3.26, un compilatore C++23 con OpenMP (GCC 14), Eigen,
Boost (container, graph, accumulators), fmt, spdlog, pybind11, GTest, Python
3.12. Con Conan: `conan install . --build=missing`; con Spack: `spack-env.yaml`.

```bash
cmake --preset release && cmake --build build     # benchmark: sempre Release
./build/test/distributed_knapsack_tests            # suite di test
./build/kpbench --test knapsackdp --numItems 1000 --capacity 1000000 \
    --maxWeight 1000 --numThreads 8 --verify --save results.json
```

Il preset `default` è Debug con i timer di dettaglio accesi: serve per
sviluppare, non per misurare. `release-timers` è Release con i timer per
oggetto, solo quando serve quel dettaglio.

Ogni esecuzione di `kpbench` stampa e salva:

- tempo totale e **fasi** (`alloc`, `first_touch`, `graph_build`, `compute`,
  `stats`, `backtrack`): gli scaling si fanno sulla fase `compute`;
- i thread **effettivamente** usati da OpenMP e il binding (`OMP_PROC_BIND`,
  `OMP_PLACES`);
- tipo di build, compilatore, flag e commit git (con `-dirty` se c'erano
  modifiche non committate);
- `valid` (la soluzione restituita è coerente) e, con `--verify`, il confronto
  con una DP seriale indipendente; esce con codice 2 se uno dei due fallisce;
- `sum_weights`: se `capacity >= sum_weights` l'istanza è banale.

## Campagne di misura

Gli script sono in `sbatch/` e girano su un nodo in esclusiva, con i thread
vincolati ai core, e rifiutano un binario che non sia Release.

| Script | Misura |
|---|---|
| `run_timing.sbatch` | tempi di tutte le varianti: banda, strong scaling fino a 80 thread, weak scaling, struttura del DAG; 1 esecuzione a vuoto + 10 misurate, `--verify` alla prima |
| `run_kpdp_perf.sbatch`, `run_kpdpdag_perf.sbatch`, `run_kpdpdag_big.sbatch` | contatori hardware con `perf stat` |
| `run_kpdp.sbatch`, `run_kpdpdag.sbatch` | prima campagna, dal driver Python (superata da `run_timing.sbatch`) |

Prova rapida in locale dell'intera campagna dei tempi, con le capacità divise
per 1000:

```bash
CAP_DIV=1000 REPEATS=1 WARMUP=0 THREADS="1 2" bash sbatch/run_timing.sbatch
```

## Riproducibilità del report

Il report si compila con `typst compile doc/main.typ`. Le figure stanno in
`doc/images/` e i dati grezzi in `doc/data/`: vanno committati insieme al
`machine.txt` della campagna che li ha prodotti, perché il report si possa
ricompilare da un clone pulito. Gli script di analisi sono in `doc/*.py`.

> ⚠️ Fra il 15/09/2026 e la revisione "Benchmark fairness" la DP classica usava
> una matrice Eigen **column-major**, circa un ordine di grandezza più lenta di
> quella attuale: le misure della classica di quel periodo, e i confronti che
> le usano, vanno rifatti. Il report segnala le sezioni coinvolte.

## Driver Python

```bash
python3 distributedknapsack --test knapsackdptiled --numItems 1000 \
    --capacity 100000 --numThreads 8 --itemBlock 10 --capBlock 10000
```

Cerca il modulo in `build/`. Imposta `OMP_NUM_THREADS` prima di caricare la
libreria e controlla che OpenMP usi davvero i thread richiesti. Genera istanze
diverse da `kpbench` a parità di seed (`random` di Python contro `std::mt19937`).
