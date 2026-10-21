#import "../functions/preamble.typ": *
#import "kp_dp_algo.typ": *

#show: style-algorithm
= Knapsack DP Classic frontier

La risoluzione classica dell'algoritmo dello zaino in versione dynamic programming è sostanzialmente l'applicazione del seguente algoritmo

#KpDp <algo-kpdp>

Questo algoritmo ha una complessità di $O(N * C)$ dove C è la capacità dello zaino, la frontiera di esecuzione lungo cui è possibile parallelizzare è sostanzialmente la linea corrente di calcolo. Visto che i calcoli effettuati sulla frontiera corrente dipendono solo da quella precedente, e che ogni esecutore accede ad una cella diversa, l'algoritmo può essere considerato parte del modello EREW PRAM (Exclusive Read, Exclusive Write).


== Knapsack DP Classic frontier: implementazione

L'implementazione più semplice di questo algoritmo è la seguente.

#figure(
  caption: "Knapsack DP Classic frontier: implementazione semplice",
  kind: "listing",
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

In questa implementazione ci sono due problemi, il primo è che l'allocazione della DP avviene come vettore contenente vettori, ogni accesso alla DP comporta l'accesso a un puntatore e questo pregiudica la località dei dati durante l'esecuzione; il secondo è che OpenMP a ogni iterazione aprirà e chiuderà i thread, con conseguente overhead. Per mitigare questo problema è necessario eliminare la chiusura e apertura dei threads e usare una struttura dati per la DP che permetta di aumentare la località dei dati, una Eigen matrix dalla libreria Eigen3 offre proprio questa funzionalità, allocando una matrice come vettore unico e offrendo operatori di accesso riga-colonna che ne semplificano la gestione.
#figure(
  caption: "Knapsack DP Classic frontier: Implementazione migliorata",
  kind: "listing",
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

La threadpool di OpenMP viene aperta nel ciclo for più esterno, in questo modo quando si procede a calcolare le righe della DP non avviene l'apertura e chiusura dei threads, invece si indica loro solo di attendere alla fine di ogni riga mediante una barriera.

== Knapsack DP Classic Frontier: Risultati

Per confrontare i risultati bisogna tenere conto di 3 parametri del problema che ne modificano profondamente il calcolo. Il primo è la capacità massima dello zaino, che cambia la dimensione della riga; il secondo è il range (e il seed di generazione) dei pesi, un peso grande richiede al processore di eseguire un salto più o meno grande nella riga precedente; l'ultimo è il numero di oggetti, che rende sostanzialmente la DP più o meno lunga, in questo caso comporta essenzialmente un incremento lineare del tempo, ma se si confrontano i tempi con la versione DAG che invece accorpa anche le linee è un parametro che è utile considerare. La granularità del problema in questo caso è dipendente dalla capacità totale, ad ogni thread viene assegnata una porzione della riga divisa in parti uguali tra threads.

=== Esecuzione dell'esperimento
L'algoritmo è stato fatto girare su una CPU Neoverse-N1, basata su architettura ARM, con 80 cores: ciascun core ha 64 KB di cache di primo livello dati (5 MB aggregati) e 80 MB di cache di secondo livello unificata. Il programma è stato compilato con GCC 14.2.0 e il preset Release di CMake, che usa i flag di ottimizzazione predefiniti (-O3).

#figure(
  image("../images/kpdp_speedup_vs_processors.png", width: 100%),
  caption: [Speedup del KnapsackDP classico per famiglia di peso e capacity.],
) <fig:kpdp-speedup>

Come si può vedere dal grafico l'algoritmo scala abbastanza bene quando si aggiungono i primi threads per poi appiattirsi sempre di più. Questo comportamento è purtroppo atteso, analizzandolo rapidamente si può infatti notare che il ciclo principale comporta sostanzialmente qualche operazione di addizione e di comparazione che verrà risolta in qualche ciclo di clock, con invece aggregate delle operazioni di load e store che possono comportare ritardi anche molto grandi.

Per capire dove viene persa la prestazione, anziché limitarci a constatare lo speedup, si possono analizzare numerose misure con lo strumento `perf` : numero di esecutori, velocità per-core e usare i contatori hardware per attribuire la perdita alla sua causa. Le misure che seguono sono state raccolte con `perf stat` eseguendo 3 repliche dell'esperimento e calcolando la mediana del contatore.

A 32 thread lo speedup si ferma a $S = T(1)/T(p) approx 7.1 - 7.7$, con efficienza $E = S/p = T(1)/(T(p) dot p) approx 0.23$ , piuttosto bassa, per indagarne le cause si possono prendere in considerazione i seguenti contatori hardware da `perf` . Il contatore `cycles` conta i cicli di clock in cui il core risulta attivo, non il lavoro completato: un load che manca in LLC e resta fermo qualche centinaio di cicli contribuisce al contatore con quei cicli pur avendo ritirato una sola istruzione. Ne segue che il totale è la somma di lavoro utile, stalli (memoria) e cicli delle istruzioni extra; in un kernel memory bound #footnote[Kernel: inteso come insieme di operazioni] come questo gran parte dei cicli è stallo anche in seriale. L'`insn ratio` è invece il rapporto fra le istruzioni eseguite dall'intero run parallelo e quelle del run seriale: il lavoro utile è per costruzione lo stesso, quindi un valore $> 1$ conta le istruzioni extra ,gestione delle barriere e attesa attiva (spinning).


Applicando il concetto di speedup (riduzione del tempo) ai cicli, si ottiene $S_"cyc" = "cycles"(1)/"cycles"(p)$: che indica di quanto si è ridotta l'esecuzione totale misurata in cicli. Un valore $S_"cyc" < 1$ significa che il run parallelo accumula più cicli del seriale; poiché però `cycles` include gli stalli, un eccesso di cicli non è per definizione attesa attiva, e per attribuirlo bisogna confrontare la crescita dei cicli con quella delle istruzioni. Nei dati i cicli crescono di $3 - 3.8 times$ mentre le istruzioni solo del +33%: se l'eccesso fosse spinning le istruzioni crescerebbero in proporzione ai cicli (un loop di attesa esegue molte istruzioni per ciclo), quindi la parte dominante sono stalli senza istruzioni ritirate , che identificano sostanzialmente delle attese sul bus da parte della CPU , e solo una frazione minore è codice di attesa sulle barriere.

La scala del fenomeno si quantifica con l'efficienza, derivata dallo speedup: a 32 thread, come visto prima, si realizza meno di un quarto dello speedup lineare ideale. Per capire dove finisce il resto possiamo porci la seguente domanda: a quale ritmo avanza, in media, un singolo esecutore durante il run? Il wall time, il tempo di esecuzione totale, si può sempre riscrivere come lavoro totale diviso throughput: se il lavoro si misura in cicli di `perf`, il throughput $Phi(p)$ è il numero di cicli che l'intero run accumula in un secondo, quindi

$
  Phi(p) = "cycles"(p)/T(p), quad T(p) = "cycles"(p)/Phi(p)
$

Il throughput di un'esecuzione ha due componenti: quanti esecutori ci sono ($p$) e quanto velocemente ciascuno accumula cicli, cioè

$
  Phi(p) = p dot "rate"(p) quad => quad "rate"(p) = Phi(p)/p = "cycles"(p)/(T(p) dot p)
$

I cicli di `perf` sono sommati su tutti i core e il tempo è quello misurato durante la stessa run, quindi $"rate"(p)$ è il numero di cicli accumulati in media da ciascun thread al secondo. Senza la normalizzazione per $p$, il throughput aggregato $Phi(p)$ cresce meccanicamente col numero di thread e non sarebbe confrontabile fra i run; dividere per $p$ isola la componente per-esecutore, rendendo i run confrontabili. Per il run seriale $"rate"(1) = "cycles"(1)/T(1)$ è a tutti gli effetti la frequenza media effettiva del core. Il rapporto fra i due run $r = "rate"(p)/"rate"(1)$ combina quindi la frequenza effettiva dei core con la quota di tempo in cui essi accumulano effettivamente cicli, e si legge come la frazione del ritmo seriale mantenuta da ciascun esecutore: un thread parcheggiato nel kernel in attesa della barriera non esegue istruzioni e quindi non accumula cicli; il suo tempo di esecuzione entra però per intero nel denominatore di $"rate"(p)$, che assegna a ogni esecutore l'intero $T(p)$ a prescindere da quanto ne abbia passato a produrre cicli: la parte di attesa "a vuoto" abbassa quindi la media. Un thread che invece fa spinning continua a eseguire istruzioni e ad accumulare cicli anche mentre attende, e il suo tempo resta coperto: per questo il parcheggio deprime $"rate"(p)$ e con esso $r$, mentre lo spinning no. Per questo esso pesa su $S_"cyc"$ e sull'insn ratio ma non su $r$, mentre il parcheggio nel kernel e il down-clocking sotto carico pesano su $r$.

A questo per sostituzione si può ricostruire lo speedup: invertendo la definizione di rate si ha $T(p) = "cycles"(p)/(p dot "rate"(p))$, che riarrangiata dà l'identità

$
  S = T(1)/T(p) = underbrace("cycles"(1)/"cycles"(p), S_"cyc") dot underbrace(p, p) dot underbrace("rate"(p)/"rate"(1), r) = S_"cyc" dot p dot r
$

I fattori si legano i questo modo: $p$ è il guadagno ideale dagli esecutori; $S_"cyc"$ è la riduzione del lavoro totale espresso in cicli (qui si intravede il peggioramento, $approx 0.27 - 0.31$: il run parallelo accumula $3 - 3.8 times$ i cicli); $r approx 0.77 - 0.84$ è l'efficienza per-core di accumulo dei cicli (frequenza effettiva e tempo parcheggiato nel kernel). Il prodotto riproduce lo speedup misurato ($approx 0.29 dot 32 dot 0.8 approx 7.4$, dentro l'intervallo $7.1 - 7.7$): la perdita di prestazioni va quindi attribuita in gran parte agli stalli di memoria aggravati dalla contesa fra thread ; visibili nel crollo dell'IPC (instructions per cycle) aggregato, cioè il rapporto istruzioni-cicli, da $approx 0.63$ a $approx 0.18 - 0.25$ a parità di lavoro e in misura minore alle istruzioni di attesa sulle barriere OpenMP (+33% di istruzioni, insn ratio). Per avere una misura di costo indipendente dalla taglia del problema si può normalizzare i cicli sul lavoro utile del kernel, che è sostanzialmente quello di riempire le celle della tabella DP: $K_c = "cycles"\/((n+1)(C+1))$ è quindi il costo hardware di una singola cella. A differenza dell'IPC, che è adimensionale e può solo qualificare il tipo di collo, $K_c$ rende confrontabili fra loro run a taglia diversa e algoritmi diversi. Il numeratore sono i cicli di `perf` e non il tempo perché l'effetto frequenza è già catturato da $r$: se $K_c$ cresce è cresciuto il costo hardware della cella, non il clock. Come si vede in @fig:kpdp-kc-vs-threads, $K_c$ è piatto rispetto alla capacity in seriale ($approx 38$ cicli/cella per entrambe le taglie), ma a 32 thread la crescita dipende leggermente dalla taglia: $approx 128$ cicli/cella a capacity $10^6$ e $approx 146$ a $10^7$, con un gradiente già marcato a 8 thread. In entrambi i casi l'IPC aggregato crolla da $approx 0.63$ a $approx 0.18 - 0.25$: le celle non costano di più perché si eseguono più istruzioni per cella, ma perché ognuna subisce più stalli. Gli stalli extra derivano dalla maggiore occupazione del bus di memoria: in parallelo $p$ flussi di letture e di write-back attraversano simultaneamente lo stesso interconnect, le code si allungano e la latenza media di ogni accesso cresce.

#figure(
  image("../images/perf_kpdp_kc_vs_threads.png", width: 100%),
  caption: [Costo per cella della DP $K_c$ in funzione del numero di thread],
) <fig:kpdp-kc-vs-threads>



=== Script di esecuzione

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



