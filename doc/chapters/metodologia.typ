#import "../functions/preamble.typ": *

= Metodologia sperimentale <sec:metodologia>

Questo capitolo fissa una volta sola le regole con cui sono prodotte le misure dei capitoli sulle implementazioni. Le campagne sono negli script di `sbatch/` e i loro dati grezzi vanno conservati in `doc/data/` insieme al file `machine.txt` della stessa esecuzione.

== Ambiente

#todo[nodo, modello di CPU, numero di core e di socket, dimensioni delle cache e topologia NUMA, dal file `machine.txt` prodotto da `run_timing.sbatch`; versione del kernel e di GCC.]

Ogni risultato salvato da `kpbench` riporta nel campo `build` tipo di build, compilatore, flag effettivi e commit git (con `-dirty` se il sorgente aveva modifiche non committate), e nel campo `omp` il numero di thread *effettivamente* usato da OpenMP e la politica di binding. Gli script rifiutano un binario che non sia Release: le misure si fanno con il preset `release` (`-O3 -DNDEBUG`, timer di dettaglio spenti). Il preset `release-timers` riaccende i timer per singolo oggetto, solo per le misure che ne hanno bisogno: aggiungono overhead nei cicli caldi.

== Controllo dell'esecuzione

- *nodo in esclusiva* (`--exclusive`): un kernel limitato dalla memoria è sensibile a chiunque altro usi la stessa banda e la stessa cache condivisa, anche se i core sono diversi;
- *binding dei thread*: `OMP_PLACES=cores` e `OMP_PROC_BIND=close` di default, un thread per core e nessuna migrazione;
- *first touch parallelo*: ogni pagina delle tabelle viene toccata la prima volta dal thread che la calcolerà, con la stessa partizione del calcolo; il costo dei page fault è misurato a parte;
- *thread da 1 a 80*, cioè fino a tutti i core del nodo.

== Ripetizioni e statistiche

Ogni configurazione viene eseguita una volta a vuoto e poi 10 volte, ciascuna in un processo separato. Si riportano la mediana e un intervallo (interquartile, o di confidenza al 95% via bootstrap), non la media e non il minimo, seguendo le indicazioni di Hoefler e Belli @hoeflerBenchmarking. I contatori hardware si raccolgono in campagne separate da quelle dei tempi: `perf stat` con più eventi dei contatori disponibili li alterna (*multiplexing*) e ne restituisce stime, perturbando le esecuzioni brevi.

== Fasi misurate

Il tempo totale di un'esecuzione comprende costi che non sono il calcolo della tabella. Ogni solver registra le proprie fasi, e tutti i grafici di scaling vanno fatti sulla fase `compute`, con il totale riportato a parte:

#figure(
  table(
    columns: 2,
    align: (left, left),
    [*fase*], [*contenuto*],
    [`alloc`], [allocazione della tabella (senza inizializzazione)],
    [`first_touch`], [primo accesso parallelo: page fault e azzeramento],
    [`graph_build`], [solo DAG: costruzione del grafo dei tile],
    [`compute`], [riempimento della tabella],
    [`stats`], [solo DAG: statistiche delle frontiere per il report],
    [`backtrack`], [ricostruzione dell'insieme di oggetti],
  ),
  caption: [Fasi registrate da ogni solver (campo `phases` dei risultati).],
)

== Metriche

- *speedup relativo* $S(p) = T(1) \/ T(p)$ della stessa variante, ed *efficienza* $E(p) = S(p) \/ p$;
- *speedup assoluto*: rispetto alla variante seriale più veloce sulla stessa istanza. Lo speedup relativo di una variante lenta in seriale può essere alto senza che la variante sia utile;
- *strong scaling* (problema fisso) e *weak scaling* (capacità proporzionale ai thread, lavoro per thread costante);
- *roofline* @williamsRoofline: `kpbench --test bandwidth` misura la banda di memoria del nodo per ogni numero di thread con una triade in stile STREAM @mccalpinStream. La DP legge due interi e ne scrive uno per cella, quindi da celle al secondo si ottengono i GB/s raggiunti e la distanza dal tetto di banda. Solo così l'affermazione "il kernel è limitato dalla memoria" diventa una misura;
- per il DAG, il limite $S_"max"(p)$ della @sec:dag-bound.

== Istanze

Pesi e valori sono interi uniformi, generati con un seed fisso. Il costo della DP è $(n+1)(C+1)$ celle qualunque siano i valori, quindi le istanze servono a fissare la dimensione del lavoro, non la difficoltà del problema. Due accorgimenti:

- se $C >= sum_i w_i$ tutti gli oggetti entrano e la risposta è banale: un solver reale lo controllerebbe prima di riempire la tabella. Ogni risultato riporta `sum_weights`, così ogni grafico può dichiarare il rapporto $C \/ sum_i w_i$;
- il peso massimo conta per la struttura delle dipendenze: la lettura "prendo" cade a distanza $w_i$ nella riga precedente, e quando $"maxWeight" > "cap"_b$ il grafo dei tile cambia forma (@sec:dag-bound).

`kpbench` (C++, `std::mt19937`) e il driver Python (`random`) generano istanze diverse a parità di seed: i confronti si fanno solo fra risultati dello stesso generatore.

== Correttezza

Ogni soluzione restituita da `kpbench` viene controllata: oggetti distinti, peso entro la capacità, somme uguali a quelle dichiarate. Con `--verify` (usato alla prima ripetizione di ogni configurazione) il valore ottimo viene confrontato con una DP seriale monodimensionale indipendente, che non condivide codice con i solver. La suite di test (`test/`) confronta tutte le varianti con lo stesso riferimento su istanze casuali con 1, 2, 3 e 7 thread, comprese capacità più piccole del numero di thread, e mette sotto carico lo scheduler topologico.
