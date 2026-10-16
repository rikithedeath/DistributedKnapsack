#import "../functions/preamble.typ": *
#import "kp_dp_algo.typ": *

#show: style-algorithm
= Knapsack DP DAG: Modello

Un modello alternativo di esecuzione per questo algoritmo è pensare di creare un Directed Acyclic Graph che rappresenti la DP invece di usare una matrice. Per farlo si consideri la ricorrenza $"Dp"_i(w)=max("Dp"_(i-1)(w), "Dp"_(i-1)(w-w_i)+v_i)$, allora si può raggiungere uno stato $(i,w)$ da $(i-1,w)$ e da $(i-1,w-w_i)$, questo insieme di archi impone sostanzialmente l'ordine di esecuzione dell'algoritmo sulle diverse celle della DP e dipende solo da $w_i$. A questo punto si possono definire le frontiere di questo DAG, in modo semplice, come l'insieme degli stati che possono essere calcolati in parallelo, dato che non dipendono da altri stati, formalmente dati due stati $u,v in F$ allora $u arrow.not v and v arrow.not u$ che formano quindi una anti chain.

Costruendo una frontiera molto semplice inizialmente, ad esempio $F_0={(0,0),(0,1),...,(0,W)}$ si possono costruire le frontiere successive utilizzando i successori dei nodi calcolati, dato un insieme di nodi completati $C$ questo insieme $"Ready"(C)={v in.not C : "Pred"(v) subset.eq C}$ forma la frontiera di nodi successiva. Se si suppone un parallelismo a livello di cella si scopre in fretta che si riproduce praticamente l'esecuzione dell'algoritmo "naive", infatti $F_1$ sarà $F_1={(1,0),(1,1),...,(1,W)}$.

Per ottenere una soluzione diversa, computazionalmente parlando, è necessario considerare un parallelismo con granularità più grossolana della cella, per farlo si può dividere la DP a blocchi (o tile). Partendo dal caso più semplice si può pensare di raggruppare porzioni della prima riga, che ospita sostanzialmente tutti i nodi della frontiera iniziale in blocchi, ora ripetendo il ragionamento precedente si può costruire nuovamente la frontiera iniziale cambiando però la condizione, per ogni blocco $B_(i,j)$ se esiste una cella $w in B_(i,j)$ che ha come dipendenza una cella di un altro blocco allora il blocco è dipendente da quel blocco. Per aumentare ulteriormente la granularità e incorporare anche gli item (quindi le righe) nel tiling si può procedere estendo la formulazione considerando una dipendenza più generica: " se i predecessori di un elemento x appartengono a un altro tile, allora il blocco attuale è dipendente dal precedente". Supponendo di costruire il DAG con i tile $G(V,E)$ con $V={(i,w)}$ e $(x,y) in E$ allora dati due blocchi A e B $A arrow B arrow.l arrow "Pred"(B) inter A eq.not emptyset$ con $"Pred"(B)=union_(x in B) "Pred"(x)$. Una volta costruito il grafo si assegnano dei livelli ad ogni tile in base alla dipendenze, ognuno di questi livelli costituisce una frontiera di esecuzione di questo DAG i cui elementi si possono risolvere in parallelo, ad ogni elemento viene poi applicata la ricorrenza originale che risolve il problema dello zaino. L'algoritmo completo viene riassunto di seguito.

#KpDpDag

Per funzionare, la procedura presuppone che i vertici all'interno del grafo siano ordinati in ordine topologico, come si vedrà nell'algoritmo successivo questa condizione viene raggiunta implicitamente inserendo i vertici in ordine row-major. L'algoritmo si articola nelle seguenti fasi: 1) Creazione del DAG. 2) Assegnamento del livello ai vertici, che corrisponde alla loro frontiera di esecuzione. 3) Esecuzione in parallelo di ogni frontiera e ritorno del risultato. La generazione del DAG segue tutte le regole espresse precedentemente e riassunte nella procedura di seguito.


#GenerateDAG

Questa procedura è piuttosto generale e dipende fortemente da come si definisce formalmente il predecessore di una cella, nel caso del knapsack una cella può avere due predecessori $"DP"[i-1][w]$ in caso si decida di non prendere l'elemento è $"DP"[i-1][w-w_i]$ nel caso in cui invece si decida di prenderlo. Per un certo elemento $i$ e un certo range di capacità $(c,c+g)$ dove g rappresenta quante celle sulla linea si vogliono raggruppare nel tile i predecessori sono due range $"DP"[i-1][c-w_i...c-w_i+g]$ e $"DP"[i-1][c...c+g]$ che rappresentano le due casistiche, a questo punto per ordinare l'esecuzione dei tile basta controllare quale dei tile contiene il range e aggiungere un arco tra quel blocco e quello corrente, si noti che i vertici vengono aggiunti in ordine row-major, questo garantisce che l'ordinamento topologico sia rispettato senza dover ricorrere a un operazione di ordinamento topologico esplicita. L'assegnamento dei livelli avviene utilizzando questa definizione in modo analogo
#AssignLevels



Questa piccola funzione implementa sostanzialmente questa ricorrenza $L(v)= cases(0 "if" "Pred"(v) = emptyset, 1+max_(x."level" in "Pred"(v)))$ che prende il massimo livello tra tutti gli elementi dei blocchi predecessori. Questa parte è piuttosto critica, siccome il livello dei predecessori deve essere assegnato prima del blocco corrente i vertici devono essere esplorati in ordine topologico, tuttavia per le condizioni precedenti questa condizione ci viene garantita dalla fase di costruzione stessa del DAG. A questo punto manca solo l'ultima fase di calcolo della funzione nei vari tile

#ComputeTile

Per ogni tile si procede con il consueto calcolo sulla DP e si ricava il risultato.

== Knapsack DP DAG: Risultati

=== Ambiente di esecuzione

Le misure condividono l'ambiente del capitolo precedente (nodo ARM Neoverse-N1 del cluster hpc4ai, `kpbench` compilato con GCC 14.2.0 in preset Release, misure `perf stat` con tempo same-run e mediana su 3 repliche). Questo consente un confronto diretto fra le due versioni: tempi e contatori provengono dalla stessa campagna.

=== L'effetto del tiling sulla località

#figure(
  image("../images/perf_vs_algorithms.png", width: 100%),
  caption: [Confronto fra KnapsackDP classico e KnapsackDPDAG (configurazione migliore) a capacity $10^6$: IPC, cicli per cella, LLC MPKI e miss LLC totali per run.],
  supplement: none,
) <fig:dag-vs-classic>

Il guadagno principale del tiling è sulla gerarchia di memoria. In seriale a capacity $10^6$ la versione DAG raggiunge un IPC di $2.2 - 2.3$ contro $approx 0.64$ del classico, con $approx 4$ miss LLC per 1000 istruzioni contro $approx 97$: le miss totali crollano da $121 times 10^6$ a $5 times 10^6$ ($24 times$ in meno) a parità di istruzioni eseguite ($approx 1.2 times 10^9$ per entrambe). Il costo per cella scende da $approx 38 - 40$ a $approx 11$ cicli: il tile mantiene il proprio working set in cache e la riga di confine del tile superiore resta residente.

Il vantaggio seriale è $approx 4 times$ (0.16 s contro 0.64 s) e *cresce* con il numero di oggetti: il costo per cella del classico degrada da 12.6 a 30.0 ns/cella passando da $n = 50$ a $n = 5000$ (la tabella, da 4 a 20 GB, peggiora cache e TLB), mentre quello del DAG resta piatto a $approx 3.2 - 3.4$ ns/cella: a $n = 5000$ il vantaggio seriale arriva a $9.4 times$.

=== La granularità dei tile

#figure(
  image("../images/perf_kpdpdag_block_grid.png", width: 100%),
  caption: [Effetto di item_block e cap_block su tempo, IPC e LLC MPKI (maxWeight=10, capacity $10^5$, 8 thread).],
  supplement: none,
) <fig:dag-block-grid>

La scelta della granularità governa il compromesso fra parallelismo esposto e overhead di schedulazione. A thread bassi vince la granularità grossa: `item_block` = 25 è ottimo a 1--2 thread, 15 a 4, 10 a 8 e 5 oltre i 16 thread (più thread richiedono più tile indipendenti). Il `cap_block` incide poco ($plus.minus 4%$ a parità di configurazione): diventa preferibile il valore grande ($10^4$) solo a capacity $10^6$ con molti thread o per maxWeight $= 1000$.

=== Scalabilità e crossover

#figure(
  image("../images/perf_kpdpdag_speedup.png", width: 100%),
  caption: [Speedup a muro $S_"wall" = T(1)\/T(p)$ del KnapsackDPDAG (configurazione migliore per $p$) per famiglia di peso e capacity.],
  supplement: none,
) <fig:dag-speedup>

La scalabilità parallela della versione DAG è limitata dal rapporto fra lavoro per tile e costo di sincronizzazione. Come si vede nella @fig:dag-speedup, a $n = 50$ lo speedup a muro si ferma a $approx 2.4 times$ e l'*insn ratio* a 32 thread sale a $4.4 times$: quasi un quarto delle istruzioni eseguite in parallelo è spinning su code e atomiche. L'esperimento di crossover (si veda il @fig:kpdp-perf-crossover nella sezione di confronto) ha verificato l'ipotesi ripetendo la misura con 20 e 100 volte più lavoro per tile:

#figure(
  table(
    columns: 6,
    [$n$], [$T_"dag"(1)$], [$T_"plain"(1)$], [vantaggio seriale], [$S_"wall"$ DAG $p=32$], [insn ratio DAG $p=32$],
    [50], [0.16 s], [0.64 s], [$4.0 times$], [$2.4 times$], [$4.4 times$],
    [1000], [3.41 s], [23.1 s], [$6.8 times$], [$7.1 times$], [$1.35 times$],
    [5000], [15.9 s], [150 s], [$9.4 times$], [$9.1 times$], [$1.25 times$],
  ),
  caption: [Crossover dello scaling: stesso problema, stessa capacity, numero di oggetti crescente (2 repliche, spread $<= 1.03$).],
  supplement: none,
) <tab:dag-crossover>

Con più lavoro fra sincronizzazioni lo speedup sale a $7.1 times$ ($n = 1000$) e $9.1 times$ ($n = 5000$) e lo spinning crolla a $approx 1.25 times$: il limite è di scala, non di algoritmo. A capacity $10^4$ (run da 1--2 ms) invece l'overhead domina e a 32 thread il tempo peggiora.

== Confronto tra la versione classica e la versione DAG

#figure(
  image("../images/perf_big_crossover.png", width: 100%),
  caption: [Speedup $S = T(1)\/T(p)$ del KnapsackDP classico e della versione tiled DAG ],
  supplement: none,
) <fig:kpdp-perf-crossover>

A capacity $10^6$ la versione DAG è più veloce del classico a *tutti* i thread count: il rapporto dei tempi passa da 0.25 a $p = 1$ a 0.78 a $p = 32$ per $n = 50$, e da 0.11 a 0.17 per $n = 5000$ (5.8 volte più veloce a 32 thread). La ragione è strutturale: il classico è memory-bound per costruzione ($approx 97$ LLC MPKI, hit $approx 0%$, IPC $approx 0.6$), e aggiungere thread non cura il problema perché i thread extra aggiungono contesa, non località ($K_c$ che cresce di $4 times$, insn ratio $+33%$). Il tiling attacca la causa e riduce il traffico DRAM di $approx 17 times$ (33 GB contro 569 GB di letture stimate a $n = 5000$, $p = 32$).

Il rovescio della medaglia appare nel regime opposto (molti oggetti, righe corte, tabella che sta in cache): lì il layout row-major del classico è già cache-ottimo, mentre il DAG paga la costruzione del grafo (centinaia di ms con decine di migliaia di tile) e l'indirezione cross-tile, e perde dal +30% al +159%. In sintesi: *quando la tabella supera la cache il DAG vince perché risolve il collo giusto (località) e il suo limite di sincronizzazione si ammette al crescere del lavoro; quando la tabella sta in cache il classico vince perché il tiling diventa puro overhead*. In entrambi i casi il limite di scaling a 32 thread non è la banda di memoria (stime $92 - 121$ GB/s crescenti senza plateau per il classico), ma la sincronizzazione.
