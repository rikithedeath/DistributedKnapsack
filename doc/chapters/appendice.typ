#import "../functions/preamble.typ": *

#heading(numbering: none)[Appendice: script delle campagne]

Gli script completi sono in `sbatch/`: `run_timing.sbatch` per i tempi (tutte le varianti, strong e weak scaling, banda, struttura del DAG), `run_kpdp_perf.sbatch`, `run_kpdpdag_perf.sbatch` e `run_kpdpdag_big.sbatch` per i contatori hardware. Tutti girano su un nodo in esclusiva, con i thread vincolati ai core, e rifiutano un binario non Release. Per riprodurre una campagna:

```bash
cmake --preset release && cmake --build build
sbatch sbatch/run_timing.sbatch          # REPEATS, THREADS, OUT sovrascrivibili
# prova rapida in locale dello stesso script, capacità divise per 1000:
CAP_DIV=1000 REPEATS=1 WARMUP=0 THREADS="1 2" bash sbatch/run_timing.sbatch
```

== Campagna di contatori della DP classica

Lo script riportato è quello della prima campagna, da cui vengono le misure della DP classica column-major. La versione attuale in `sbatch/` aggiunge nodo in esclusiva, binding dei thread, controllo che il binario sia Release e ripetizioni configurabili.

La campagna di misure è stata eseguita sul cluster mercurio sulla partizione epito mediante il seguente script Slurm, che per ogni combinazione di parametri ripete la misura per un numero fissato di repliche: in ogni replica `perf stat` raccoglie i contatori hardware in formato CSV mentre il tempo del solver viene letto dallo stdout di `kpbench` della stessa esecuzione, così che contatori e tempi appartengano sempre allo stesso run; le righe vengono poi appese a un unico CSV di riepilogo con i metadati del run (la colonna `time_s` è vuota solo se la riga del tempo non è stata riconosciuta). Lo script verifica preventivamente che `perf` possa usare gli eventi richiesti e abortisce in caso contrario.

#figure(
  caption: "sbatch della campagna di misure sul DP classico",
  kind: "listing",
  sourcecode[
    ```bash
    #!/bin/bash
    #SBATCH --job-name=kpdp_perf
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=32
    #SBATCH --time=00:30:00
    #SBATCH --output=perf_kpdp.out
    #SBATCH --error=perf_kpdp.err
    #SBATCH --mem=32G
    set -euo pipefail
    SEED=42
    NUM_ITEMS=50
    BIN="./build/kpbench"
    REPEATS=3
    TEST="knapsackdp"
    OUT_DIR="perf_kpdp_out"
    SUMMARY="$OUT_DIR/perf_summary.csv"
    SUMMARY_HEADER="max_weight,capacity,threads,tag,
    value,unit,event,runtime_ns,runtime_pct,metric_value,metric_unit,repeat,time_s"

    PERF_EVENTS="cycles,instructions,cache-references,cache-misses,
    branch-misses,L1-dcache-loads,L1-dcache-load-misses,ll_cache_rd,
    ll_cache_miss_rd,dTLB-load-misses,iTLB-load-misses"

    for max_weight in 10 1000; do
      for threads in 1 8 32; do
        for cap in 1000000 10000000; do
          tag="maxw${max_weight}_cap${cap}_t${threads}"
          export OMP_NUM_THREADS="$threads"

          for rep in $(seq 1 "$REPEATS"); do
            RUN_OUT=$(perf stat -x , -e "$PERF_EVENTS" \
              -o "$OUT_DIR/perf_${tag}_r${rep}.csv" \
              "$BIN" \
                --test "$TEST" \
                --capacity "$cap" \
                --numItems "$NUM_ITEMS" \
                --seed "$SEED" \
                --numThreads "$threads" \
                --maxWeight "$max_weight" \
                --save "")

            time_s=$(printf '%s\n' "$RUN_OUT" | sed -n "s/^${TEST}: \([0-9.]*\)s .*/\1/p")
            [ -s "$SUMMARY" ] || echo "$SUMMARY_HEADER" > "$SUMMARY"
            awk -F, -v OFS=, -v mw="$max_weight" -v c="$cap" -v p="$threads" -v tag="$tag" \
              -v rep="$rep" -v t="$time_s" \
              '!/^#/ && NF >= 5 {
                 mv = (NF >= 6) ? $6 : ""
                 mu = (NF >= 7) ? $7 : ""
                 print mw, c, p, tag, $1, $2, $3, $4, $5, mv, mu, rep, t
               }' \
              "$OUT_DIR/perf_${tag}_r${rep}.csv" >> "$SUMMARY"
          done

          echo ""
        done
      done
    done

    ```
  ],
) <lst:kpdp-perf-sbatch>



