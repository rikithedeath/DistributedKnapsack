#import "../functions/preamble.typ": *
#import "kp_dp_algo.typ": *

#show: style-algorithm
= Knapsack DP Classic frontier

La risoluzione classica dell'algoritmo dello zaino in versione dynamic programming è sostanzialmente l'applicazione del seguente algoritmo

#KpDp

Questo algoritmo ha una complessità di $O(N * C)$ dove C è la capacità dello zaino, la frontiera di esecuzione lungo cui è possibile parallelizzare è sostanzialmente la linea corrente di calcolo. A prima vista si potrebbe pensare che l'algoritmo non stia garantendo correttamente l'ordine di esecuzione, ad un ispezione più accurata invece si può notare che il calcolo di ogni linea $i$ ha come unica dipendenza che il calcolo sulla linea $i-1$ sia stato eseguito e che tutti gli esecutori abbiano terminato ogni operazione sulla linea $i-1$ rendendo di fatto possibile la sua esecuzione su EREW PRAM (Exclusive Read, Exclusive Write) visto che tutti gli accessi in lettura avvengono solo sulla linea precedente ad opera di un singolo esecutore e in scrittura solo sulla linea corrente, anche qui ad opera di un singolo esecutore.


== Knapsack DP Classic frontier: implementazione

L'implementazione più semplice di questo algoritmo è la seguente.

#figure(
  caption: "Knapsack DP Classic frontier: implementazione semplice",
  kind: "listing",
  supplement: none,
  sourcecode[
    ```cpp
    int n = weights.size();
    std::vector<std::vector<int>> dp(n + 1, std::vector<int>(capacity + 1, 0));
    // Build the dp table
    for (int i = 1; i <= n; ++i) {
        #pragma omp parallel for
        for (int w = 0; w <= capacity; ++w) {
            if (weights[i - 1] <= w) {
                dp[i][w] = std::max(dp[i - 1][w], dp[i - 1][w - weights[i - 1]] + values[i - 1]);
            } else {
                dp[i][w] = dp[i - 1][w];
            }
        }
    }
    ```
  ],
)

In questa implementazione ci sono due problemi, il primo è che l'allocazione della DP avviene come vettore contenente vettori, ogni accesso alla DP comporta l'accesso a un puntatore e questo pregiudica la località dei dati durante l'esecuzione; il secondo è che OpenMP a ogni iterazione aprirà e chiuderà i thread, con conseguente overhead. Per ovviare al problema si usa la seguente implementazione

#figure(
  caption: "Knapsack DP Classic frontier: Implementazione migliorata",
  kind: "listing",
  supplement: none,
  sourcecode[
    ```cpp
    int n = weights.size();
    	Eigen::MatrixX<int> dp{};
    	dp.resize(n + 1, capacity + 1);
    	dp.setZero();
    #pragma omp parallel
    	for (int i = 1; i <= n; ++i)
    	{
    		int tid = omp_get_thread_num();
    		int nt = omp_get_num_threads();
    		int begin = std::ceil(tid * capacity / nt);
    		int end = std::min<int>(capacity, std::ceil((tid + 1) * capacity / nt));
    		for (int w = begin; w <= end; ++w)
    		{
    			if (weights[i - 1] <= w)
    			{
    				dp(i, w) = std::max(dp(i - 1, w), dp(i - 1, w - weights[i - 1]) + values[i - 1]);
    			}
    			else
    			{
    				dp(i, w) = dp(i - 1, w);
    			}
    		}
    #pragma omp barrier
    	}
    ```
  ],
)

In questa implementazione si usa una Eigen matrix al posto di un vettore di vettori per rappresentare la DP, che viene allocata in row major order, il vantaggio è che questo oggetto garantisce accessi facili da leggere e un allocazione contigua in memoria della tabella, massimizzando in questo modo la località dei dati. La threadpool di OpenMP viene aperta nel ciclo for più esterno, in questo modo quando si procede a calcolare le righe della DP non avviene l'apertura e chiusura dei threads, invece si indica loro solo di attendere alla fine di ogni riga mediante una barriera.

== Knapsack DP Classic Frontier: Risultati

Per confrontare i risultati bisogna tenere conto di 3 parametri del problema che ne modificano profondamente il calcolo. Il primo è la capacità massima dello zaino, che cambia la dimensione della riga; il secondo è il range (e il seed di generazione) dei pesi, un peso grande richiede al processore di eseguire un salto più o meno grande, che potrebbe comportare un cache miss; l'ultimo è il numero di oggetti, che rende sostanzialmente la DP più o meno lunga, in questo caso comporta essenzialmente un incremento lineare del tempo, ma se si confrontano i tempi con la versione DAG che invece accorpa anche le linee è un parametro che è utile considerare.

=== Ambiente di esecuzione
L'algoritmo è stato fatto girare su una CPU Neoverse-N1, basata su architettura ARM, possiede 80 cores; una cache di primo livello di 5 MB e una cache di secondo livello di 80 MB. Il programma è stato compilato con GCC 14.2.0 e il preset Release di CMake , che automaticamente include -Os.

// includi figura qui

#figure(
  image("../images/kpdp_speedup_vs_processors.png", width: 100%),
  caption: [Speedup $S = T(1)\/T(p)$ del KnapsackDP classico per famiglia di peso e capacity.],
  supplement: none,
) <fig:kpdp-speedup>

Come si può vedere dal grafico l'algoritmo scala abbastanza bene quando si aggiungono i primi threads per poi appiattirsi sempre di più. Questo comportamento è atteso poichè l'algoritmo è decisamente memory bound, analizzandolo rapidamente si può infatti notare che il ciclo principale comporta sostanzialmente qualche operazione di addizione e di comparazione che verrà risolta in qualche ciclo di clock, con invece aggregate delle operazioni di load e store che possono comportare ritardi anche molto grandi.

Le misure che seguono sono state raccolte con `perf stat` eseguendo 3 repliche dell'esperimento e calcolando la mediana del contatore.

Per capire *dove* viene persa la prestazione, anziché limitarci a constatare lo speedup, il paragrafo seguente scompone il tempo di esecuzione nei fattori che lo determinano divisione del lavoro, numero di esecutori, velocità per-core e usa i contatori hardware per attribuire la perdita alla sua causa: sincronizzazione o memoria.

A 32 thread lo speedup si ferma a $S approx 7.1 - 7.7$ (circa il 23% di efficienza). La scomposizione $S = S_"cyc" dot p dot r$ vede in $S_"cyc" = "cycles"(1)\/"cycles"(p)$ la riduzione del numero totale di cicli e in $r$ il rapporto di ciclo per-core. Il fattore $r$ è una grandezza derivata definita come rapporto fra i tassi di ciclo per-core $r(p) = "rate"(p)\/"rate"(1)$, dove $"rate"(p) = "cycles"(p)\/(T(p) dot p)$ è il numero di cicli accumulati in media da ciascun thread al secondo (i cicli di `perf` sono sommati su tutti i core e il tempo è quello same-run della stessa misura). $r$ combina quindi la frequenza effettiva dei core con la quota di tempo in cui i core effettivamente accumulano cicli: un thread parcheggiato nel kernel in attesa della barriera non ne accumula, mentre uno che fa spinning sì per questo lo spinning pesa su $S_"cyc"$ e sull'*insn ratio* e non su $r$. L'applicazione della scomposizione ai dati mostra che i cicli totali crescono di un fattore $3 - 3.8 times$ ($S_"cyc" approx 0.27 - 0.31$) mentre l'efficienza per-core resta $r approx 0.77 - 0.84$: i thread extra spendono cicli in attesa e spinning sulle barriere OpenMP, fino al +33% di istruzioni eseguite rispetto alla versione seriale (visibile ai contatori con l'indice *insn ratio*). Per dare una misura di costo indipendente dalla taglia del problema si può modellareil costo per cella nella tabella DP: il denominatore $(n+1)(C+1)$ è il numero esatto di celle della DP (le righe $i = 0..n$ includono il caso base "zaino vuoto", le colonne $w = 0..C$ il caso "capacità residua nulla") e poiché riempire ogni cella è l'unico lavoro utile del kernel, dividere i cicli totali per le celle normalizza l'overhead rispetto al lavoro produttivo. Il numeratore sono i cicli di `perf` e non il tempo perché i cicli misurano lavoro e stalli ma non la frequenza: se $K_c$ cresce è cresciuto il costo hardware della cella, non il clock, l'effetto frequenza è già catturato da $r$. In questo modo $K_c = "cycles"\/((n+1)(C+1))$ è piatto rispetto alla capacity ma cresce da $approx 38$ a $approx 146 - 149$ cicli/cella passando da 1 a 32 thread, e l'IPC aggregato crolla da $approx 0.63$ a $approx 0.18 - 0.25$; l'IPC (instructions per cycle) è qui aggregato su tutta l'esecuzione, cioè $"IPC" = "instr."\/"cycles"$ con entrambi i contatori sommati su tutti i core e sull'intero run, quindi include nel rapporto sia il lavoro utile sia lo spinning delle attese.

=== Osservazioni sui cache miss

I contatori hardware confermano che il collo di bottiglia è la gerarchia di memoria. L'hit ratio sull'ultimo livello di cache è 0--3%: la tabella DP (40 MB a $C = 10^6$, 400 MB a $C = 10^7$) eccede di gran lunga la cache e il layout row-major usa ogni cache line una volta sola prima di espellerla; la riga $i-1$, unica dipendenza del calcolo, viene riletta quando è già stata espulsa. Il risultato è $approx 90 - 100$ miss LLC per 1000 istruzioni (LLC MPKI) con un IPC di $approx 0.6$: il core passa la maggior parte del tempo ad aspettare la memoria.

L'hit ratio L1 resta invece alto (83--90%) perché ogni cella viene scritta e riletta a breve distanza (dipendenza $"dp"[i-1][w-1]$), e i branch miss sono trascurabili ($< 0.02%$ delle istruzioni): il limite non è il flusso di controllo ma il traffico verso la memoria. Da notare che il tasso di miss sulla TLB dati resta piatto ($approx 3 - 5$ MPKI) su tutto l'intervallo di capacity: anche con la tabella da 400 MB il page-walk non è ciò che degrada le prestazioni, e il degrado di $K_c$ col numero di thread è contesa e sincronizzazione, non TLB. Curiosamente il LLC MPKI *diminuisce* con i thread ($97 arrow 74$ a maxWeight=10): con più core che lavorano su porzioni di riga in parallelo, le cache line di confine vengono riutilizzate da più core prima di essere espulse --- meno miss pro-capite, ma più contesa.

Infine, la stima della banda DRAM effettiva (letture ricostruite dai miss LLC più una scrittura di 4 B per cella, divise per il tempo del solver) passa da $approx 12$ GB/s in seriale a $approx 92 - 121$ GB/s a 32 thread, con curve ancora crescenti e senza plateau: neanche a 32 thread la banda di memoria è il muro. Il limite di scaling di questa versione è quindi la sincronizzazione più che la memoria stessa; il confronto con la versione tiled DAG --- che riduce i miss LLC di $18 - 28 times$ mantenendo un costo per cella costante al crescere del numero di oggetti --- è discusso nel capitolo dedicato.

=== Script di esecuzione

La campagna di misure è stata eseguita sul cluster mercurio sulla partizione epito mediante il seguente script Slurm, che per ogni combinazione di parametri ripete la misura per un numero fissato di repliche: in ogni replica `perf stat` raccoglie i contatori hardware in formato CSV mentre il tempo del solver viene letto dallo stdout di `kpbench` della stessa esecuzione, così che contatori e tempi appartengano sempre allo stesso run; le righe vengono poi appese a un unico CSV di riepilogo con i metadati del run (la colonna `time_s` è vuota solo se la riga del tempo non è stata riconosciuta). Lo script verifica preventivamente che `perf` possa usare gli eventi richiesti e abortisce in caso contrario.

#figure(
  caption: "sbatch della campagna di misure sul DP classico",
  kind: "listing",
  supplement: none,
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
    SUMMARY_HEADER="max_weight,capacity,threads,tag,value,unit,event,runtime_ns,runtime_pct,metric_value,metric_unit,repeat,time_s"
    PERF_EVENTS="cycles,instructions,cache-references,cache-misses,branch-misses,L1-dcache-loads,L1-dcache-load-misses,ll_cache_rd,ll_cache_miss_rd,dTLB-load-misses,iTLB-load-misses"

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



