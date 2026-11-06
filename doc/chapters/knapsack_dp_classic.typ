#import "../functions/preamble.typ": *
#import "kp_dp_algo.typ": *

#show: style-algorithm
= DP classica: parallelismo sulla riga

La risoluzione classica dell'algoritmo dello zaino in versione dynamic programming è sostanzialmente l'applicazione del seguente algoritmo

#KpDp <algo-kpdp>

Questo algoritmo ha una complessità di $O(n dot C)$, dove $C$ è la capacità dello zaino: è *pseudo-polinomiale*, polinomiale nel valore di $C$ ma esponenziale nella sua dimensione in bit ($log_2 C$), ed è il motivo per cui il problema resta NP-difficile pur ammettendo questa soluzione esatta. La frontiera di esecuzione lungo cui è possibile parallelizzare è la riga corrente: le celle della riga $i$ dipendono solo dalla riga $i-1$ e ogni esecutore scrive celle diverse. Le letture invece *non* sono esclusive: la cella $"DP"[i-1][x]$ è letta sia dalla cella $w = x$ (caso "non prendo") sia dalla cella $w = x + w_i$ (caso "prendo"), che in generale appartengono a esecutori diversi. Il modello di riferimento è quindi la CREW PRAM (Concurrent Read, Exclusive Write): con $C+1$ processori l'algoritmo esegue $n$ passi paralleli, cioè lavoro $O(n C)$ e profondità $O(n)$.


== Implementazione

L'implementazione più semplice di questo algoritmo è la seguente.

#figure(
  caption: "DP classica: implementazione semplice",
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

In questa implementazione il problema reale è uno solo: OpenMP apre e chiude il team di thread a ogni riga, con un overhead che si paga $n$ volte. L'allocazione come vettore di vettori, invece, *non* pregiudica la località: ogni riga è un blocco contiguo e il ciclo interno scorre proprio lungo la riga, per cui l'indirezione costa un accesso al puntatore per riga, non per cella.

Una matrice Eigen alloca l'intera tabella in un blocco unico e ne semplifica l'indicizzazione, ma va dichiarata con l'ordine giusto. L'ordine predefinito di Eigen è *per colonne* (column-major): con il ciclo interno su $w$, due celle consecutive $"DP"(i, w)$ e $"DP"(i, w+1)$ distano $(n+1) dot 4$ byte, cioè 204 B a $n = 50$ e 20 KB a $n = 5000$. Ogni cella tocca allora una linea di cache diversa e, per $n$ grande, una pagina diversa. La prima versione della matrice usava proprio l'ordine predefinito; la versione corretta la dichiara row-major, così che ogni riga torni a essere un flusso contiguo:

#figure(
  caption: "DP classica: implementazione corretta (estratto di `knapsackdp.cpp`)",
  kind: "listing",
  sourcecode[
    ```cpp
    using DpTable = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    DpTable dp;
    dp.resize(n + 1, capacity + 1);   // nessun setZero() seriale
    #pragma omp parallel
    {
        // fetta semiaperta [begin, end) della riga, in 64 bit
        const auto [begin, end] = kp::static_block(capacity + 1, omp_get_thread_num(),
                                                   omp_get_num_threads());
        for (int i = 1; i <= n; ++i) {
            kp::dp_row_segment(dp.row(i - 1).data(), dp.row(i).data(),
                               begin, end, weights[i - 1], values[i - 1]);
            #pragma omp barrier
        }
    }
    ```
  ],
) <lst:kpdp-row-major>

Il team di OpenMP viene aperto fuori dal ciclo sulle righe: i thread non vengono creati e distrutti a ogni riga, ma si attendono a vicenda con una barriera alla fine di ognuna, necessaria perché la riga $i+1$ legge la riga $i$ in colonne arbitrarie ($w - w_(i+1)$). Tre dettagli della versione corretta contano per le misure:

- *fette disgiunte*: ogni thread riceve l'intervallo semiaperto $[b_t, e_t)$ calcolato da `kp::static_block` in aritmetica a 64 bit. La prima versione usava estremi inclusi su entrambi i lati, per cui la colonna di confine fra due thread veniva calcolata due volte (una data race, benigna nel valore ma formalmente comportamento indefinito), e il prodotto `tid * capacity` in `int` andava in overflow oltre $approx 6.7 dot 10^7$ con 32 thread;
- *first touch parallelo*: la tabella non viene azzerata in seriale ma toccata la prima volta da ciascun thread nella propria fetta, con la stessa partizione del calcolo. Ogni pagina finisce così sul nodo NUMA del thread che la userà, e il costo dei page fault è misurato come fase a sé (`first_touch`) invece di gonfiare il tempo del kernel;
- *kernel comune*: il ciclo interno (`kp::dp_row_segment`) è spezzato in $w = w_i$, così la parte calda non ha salti dipendenti dai dati e il compilatore la vettorizza. Lo stesso kernel è usato dalle varianti a tile e dal DAG, per cui una differenza di tempo fra le varianti può venire solo dall'ordine di visita, dal layout o dalla sincronizzazione.

=== Variante con due righe e tabella delle decisioni

La tabella completa serve solo alla ricostruzione della soluzione. La variante `knapsackdpbits` tiene due sole righe di valori, scambiate a ogni oggetto, e registra per ogni cella un bit "l'oggetto $i$ è stato preso alla capacità $w$" (vero se e solo se $"DP"[i][w] > "DP"[i-1][w]$, la stessa regola della ricostruzione classica, per cui le due versioni restituiscono lo stesso insieme). La memoria scende da 4 byte a 1/8 di byte per cella, cioè 32 volte, e la tabella da 20 GB a $n = 5000$, $C = 10^6$ diventa di 625 MB. Ogni thread possiede parole intere da 64 bit della tabella dei bit, così due thread non scrivono mai la stessa parola. È il riferimento classico più forte con cui una versione a tile deve confrontarsi.

== Risultati

#nota(titolo: "Misure da rifare")[Tutti i tempi, i contatori e i valori di $K_c$ di questa sezione sono stati raccolti con la prima versione della matrice, *column-major*. Descrivono quindi il costo di una visita a passo $(n+1) dot 4$ byte, non quello dell'algoritmo: la tabella row-major legge e scrive ogni riga come un flusso contiguo. L'interpretazione "memory-bound per costruzione" va riletta in questa luce (@sec:kpdp-layout), e le misure vanno rifatte con `sbatch/run_timing.sbatch` e con i `run_*_perf.sbatch` aggiornati (@sec:metodologia).]


Per confrontare i risultati bisogna tenere conto di 3 parametri del problema che ne modificano profondamente il calcolo. Il primo è la capacità massima dello zaino, che cambia la dimensione della riga; il secondo è il range (e il seed di generazione) dei pesi, un peso grande richiede al processore di eseguire un salto più o meno grande nella riga precedente; l'ultimo è il numero di oggetti, che rende sostanzialmente la DP più o meno lunga, in questo caso comporta essenzialmente un incremento lineare del tempo, ma se si confrontano i tempi con la versione DAG che invece accorpa anche le linee è un parametro che è utile considerare. La granularità del problema in questo caso è dipendente dalla capacità totale, ad ogni thread viene assegnata una porzione della riga divisa in parti uguali tra threads.

=== Esecuzione dell'esperimento
L'algoritmo è stato fatto girare su un nodo della partizione epito del cluster mercurio, con CPU Arm Neoverse-N1 a 80 core. Nel Neoverse-N1 la cache dati di primo livello è da 64 KB e la cache di secondo livello è *privata* di ciascun core (non unificata); il livello condiviso dipende dal SoC. #todo[dimensioni di L2 e dell'ultimo livello, e topologia NUMA, dal file `machine.txt` della campagna `run_timing.sbatch`.] Il programma risulta compilato con GCC 14.2.0 nel preset Release di CMake (`-O3 -DNDEBUG`). #todo[verificare: gli script di allora indicavano il preset `default`, che è Debug; ogni risultato nuovo riporta tipo di build, compilatore, flag e commit nel campo `build`.]

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



=== Lettura delle misure alla luce del layout <sec:kpdp-layout>

Tre segnali delle misure precedenti sono la firma di una visita a passo lungo più che di un limite dell'algoritmo:

- *il costo per cella cresce con $n$*: da 12,6 a 30 ns per cella passando da $n = 50$ a $n = 5000$. Con la tabella row-major il ciclo interno legge e scrive righe contigue e non dipende da $n$; con quella column-major il passo fra celle consecutive è proporzionale a $n+1$, e da $n approx 1000$ in su ogni cella cade in una pagina diversa (TLB);
- *$approx 38$ cicli per cella e hit sull'ultimo livello di cache allo 0--3%*: con un passo di 204 byte ogni accesso porta in cache una linea da 64 byte di cui si usa un solo intero;
- *lo speedup del DAG in seriale* ($4 - 9.4 times$, @tab:dag-crossover): il DAG memorizza ogni tile row-major, quindi il confronto misura il layout prima del tiling.

#nota(titolo: "Verifica preliminare")[Su una macchina di sviluppo x86-64 (una VM con 4 vCPU, non il nodo del cluster), a $n = 50$, $C = 10^6$, maxWeight = 10, mediana di 3 esecuzioni e tempo totale del solver: la versione column-major impiega 0,589 s in seriale e quella row-major 0,045 s ($approx 13 times$); il DAG con $("item"_b, "cap"_b) = (10, 10^4)$ ne impiega 0,099 s. Con la tabella row-major la classica è *più veloce* del DAG già in seriale. Il numero vale solo per quella macchina e serve a mostrare l'ordine di grandezza dell'effetto: le conclusioni del confronto vanno tratte dalle nuove misure sul nodo del cluster.]

Lo script della campagna di contatori è riportato in appendice (@lst:kpdp-perf-sbatch).
