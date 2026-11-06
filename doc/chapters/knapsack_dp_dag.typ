#import "../functions/preamble.typ": *
#import "kp_dp_algo.typ": *

#show: style-algorithm
= DP a tile su un grafo delle dipendenze

Un modello alternativo di esecuzione al knapsack classico è pensare di creare un Directed Acyclic Graph che rappresenti il flusso di esecuzione invece di operare direttamente sulla linea come frontiera. Per farlo si consideri la ricorrenza $"Dp"_i(w)=max("Dp"_(i-1)(w), "Dp"_(i-1)(w-w_i)+v_i)$, allora si può raggiungere uno stato $(i,w)$ da $(i-1,w)$ e da $(i-1,w-w_i)$, questo insieme di archi impone sostanzialmente l'ordine di esecuzione dell'algoritmo sulle diverse celle della DP e dipende solo da $w_i$. A questo punto si possono definire le frontiere di questo DAG, in modo semplice, come l'insieme degli stati che possono essere calcolati in parallelo, dato che non dipendono da altri stati, formalmente dati due stati $u,v in F$ allora $u arrow.not v and v arrow.not u$ che formano quindi una anti chain.

Costruendo una frontiera molto semplice inizialmente, ad esempio $F_0={(0,0),(0,1),...,(0,W)}$ si possono costruire le frontiere successive utilizzando i successori dei nodi calcolati, dato un insieme di nodi completati $C$ questo insieme $"Ready"(C)={v in.not C : "Pred"(v) subset.eq C}$ forma la frontiera di nodi successiva. Se si suppone un parallelismo a livello di cella si ottiene che si sta riproducendo il flusso dell'algoritmo classico, infatti $F_1$ sarà $F_1={(1,0),(1,1),...,(1,W)}$.

Per generalizzare il comportamento delle frontiere si può considerare un parallelismo con granularità più grossolana della cella, per farlo si può dividere la DP a blocchi (o tile). Partendo dal caso più semplice si può pensare di raggruppare porzioni della prima riga, che ospita sostanzialmente tutti i nodi della frontiera iniziale in blocchi, ora ripetendo il ragionamento precedente si può costruire nuovamente la frontiera iniziale cambiando però la condizione, per ogni blocco $B_(i,j)$ se esiste una cella $w in B_(i,j)$ che ha come dipendenza una cella di un altro blocco allora il blocco è dipendente da quel blocco. Per aumentare ulteriormente la granularità e incorporare anche gli item (quindi le righe) nel tiling si può procedere estendo la formulazione considerando una dipendenza più generica: " se i predecessori di un elemento x appartengono a un altro tile, allora il blocco attuale è dipendente dal precedente". Supponendo di costruire il DAG con i tile $G(V,E)$ con $V={(i,w)}$ e $(x,y) in E$ allora dati due blocchi A e B $A arrow B arrow.l arrow "Pred"(B) inter A eq.not emptyset$ con $"Pred"(B)=union_(x in B) "Pred"(x)$. Una volta costruito il grafo si assegnano dei livelli ad ogni tile in base alla dipendenze, ognuno di questi livelli costituisce una frontiera di esecuzione di questo DAG i cui elementi si possono risolvere in parallelo, ad ogni elemento viene poi applicata la ricorrenza originale che risolve il problema dello zaino. L'algoritmo completo viene riassunto di seguito.

#KpDpDag

Per funzionare, la procedura presuppone che i vertici all'interno del grafo siano ordinati in ordine topologico, come si vedrà nell'algoritmo successivo questa condizione viene raggiunta implicitamente inserendo i vertici in ordine row-major. L'algoritmo si articola nelle seguenti fasi: 1) Creazione del DAG. 2) Assegnamento del livello ai vertici, che corrisponde alla loro frontiera di esecuzione. 3) Esecuzione in parallelo di ogni frontiera e ritorno del risultato. La generazione del DAG segue tutte le regole espresse precedentemente e riassunte nella procedura di seguito.


#GenerateDAG

Questa procedura è piuttosto generale e dipende fortemente da come si definisce formalmente il predecessore di una cella, nel caso del knapsack una cella può avere due predecessori $"DP"[i-1][w]$ in caso si decida di non prendere l'elemento è $"DP"[i-1][w-w_i]$ nel caso in cui invece si decida di prenderlo. Per un certo elemento $i$ e un certo range di capacità $(c,c+g)$ dove g rappresenta quante celle sulla linea si vogliono raggruppare nel tile i predecessori sono due range $"DP"[i-1][c-w_i...c-w_i+g]$ e $"DP"[i-1][c...c+g]$ che rappresentano le due casistiche, a questo punto per ordinare l'esecuzione dei tile basta controllare quale dei tile contiene il range e aggiungere un arco tra quel blocco e quello corrente, si noti che i vertici vengono aggiunti in ordine row-major, questo garantisce che l'ordinamento topologico sia rispettato senza dover ricorrere a un operazione di ordinamento topologico esplicita. L'assegnamento dei livelli avviene utilizzando questa definizione in modo analogo
#AssignLevels



Questa piccola funzione implementa sostanzialmente questa ricorrenza $L(v)= cases(0 "if" "Pred"(v) = emptyset, 1+max_(x."level" in "Pred"(v)))$ che prende il massimo livello tra tutti gli elementi dei blocchi predecessori. Questa parte è piuttosto critica, siccome il livello dei predecessori deve essere assegnato prima del blocco corrente i vertici devono essere esplorati in ordine topologico, tuttavia per le condizioni precedenti questa condizione ci viene garantita dalla fase di costruzione stessa del DAG. A questo punto manca solo l'ultima fase di calcolo della funzione nei vari tile

#ComputeTile

Per ogni tile si procede con il consueto calcolo sulla DP e si ricava il risultato.

== Knapsack DP DAG: implementazione

L'implementazione mantiene una corrispondenza quasi biunivoca con il modello: i tile sono i vertici di un grafo `boost::adjacency_list` bidirezionale e ogni vertice porta appeso un `NodeData` che descrive il sottoblocco della tabella di sua competenza. Il tiling è parametrizzato da `item_block` e `cap_block`, che accorpano rispettivamente le righe (gli item) e le colonne (le capacità) della DP: la griglia ha $n_b = ceil(n\/"item"_b)$ blocchi di item e $n_q = ceil((C+1)\/"cap"_b)$ blocchi di capacità, e poiché i vertici vengono inseriti in ordine row-major il descrittore del tile $(b, q)$ è semplicemente $b dot n_q + q$. Il blocco DP è una matrice Eigen row-major con una riga in più rispetto agli item locali del tile: la riga 0 non viene calcolata ma ereditata dal tile sopra (riga di confine), mentre per ogni item locale `dep_ranges` registra l'intervallo di capacità letto nel caso "take", cioè il range del tile shiftato di $w_i$.

#figure(
  caption: "Knapsack DP DAG: strutture dati del grafo",
  kind: "listing",
  sourcecode[
    ```cpp
    using BlockMatrix = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    struct NodeData
    {
        int b, q;          // coordinate del tile nella griglia
        int rows;          // item locali + 1 (riga 0 = riga di confine)
        int width;         // colonne di capacità del tile
        int level;         // livello = frontiera di esecuzione
        int lo, hi;        // intervallo di capacità posseduto
        std::vector<std::pair<int, int>> dep_ranges; // dipendenza per item locale
        BlockMatrix block; // blocco DP, dimensioni rows x width
    };

    using Graph = boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS,
        boost::property<boost::vertex_index_t, int, NodeData>>;
    ```
  ],
)

La costruzione del grafo (`build_graph`) procede in due passaggi. Nel primo vengono creati tutti i vertici in ordine row-major, allocando per ciascuno il blocco e calcolando gli intervalli di dipendenza descritti sopra. Nel secondo si aggiungono gli archi: oltre all'arco col tile direttamente sopra $(b-1, q)$, che copre la dipendenza del caso "skip" ($"DP"[i-1][w]$), per ogni item locale con intervallo non vuoto si aggiungono archi dai blocchi di capacità che intersecano l'intervallo shiftato, sia sulla riga di tile sopra sia sulla stessa riga a sinistra: sono i predecessori del caso "take" ($"DP"[i-1][w - w_i]$, che può trovarsi nel tile sopra o in uno a sinistra della stessa riga). Ogni arco codifica esattamente la regola dell'algoritmo :"se un predecessore di una cella del tile vive in un altro tile", il tile dipende da quel tile" e l'inserimento row-major dei vertici garantisce l'ordinamento topologico senza alcuna operazione esplicita, come richiesto dalla procedura.

L'assegnamento dei livelli (`compute_levels`) è un singolo passaggio sui vertici in ordine row-major: quando un tile viene visitato tutti i suoi predecessori hanno descrittore minore per cui il livello è già assegnato. La risoluzione (`solve_dag`) smista i tile in livelli ed esegue le frontiere una dopo l'altra, con parallelismo interno al livello:

#figure(
  caption: "Knapsack DP DAG: esecuzione a frontiere",
  kind: "listing",
  sourcecode[
    ```cpp
    compute_levels(g); 
    int max_level = 0;
    for (int b = 0; b < nb; ++b)
        for (int q = 0; q < nq; ++q)
            max_level = std::max(max_level, g[b * nq + q].level);

    std::vector<std::vector<int>> by_level(max_level + 1);
    for (int b = 0; b < nb; ++b)
        for (int q = 0; q < nq; ++q)
            by_level[g[b * nq + q].level].push_back(b * nq + q);

    for (int L = 0; L <= max_level; ++L)
    {
        const std::vector<int> &tiles = by_level[L];
        // il calcolo sul livello successivo deve avvenire solo quando concluso quello corrente
        #pragma omp parallel for
        for (std::size_t t = 0; t < tiles.size(); ++t)
            compute_tile(g, tiles[t], weights, values, item_block, cap_block);
    }
    ```
  ],
)

La barriera implicita a fine `parallel for` realizza esattamente la semantica di $"Ready"(C)$ del modello: un tile del livello $L+1$ non comincia prima che tutti quelli del livello $L$ abbiano scritto il proprio blocco. Il valore ottimo si legge infine nell'angolo in basso a destra dell'ultimo tile, `block(rows - 1, capacity - lo)`.

Il calcolo di un singolo tile (`compute_tile`) è la ricorrenza classica confinata al sottoblocco. La riga 0 viene copiata dal tile sopra: per ogni capacità $w$ dell'intervallo si risale al tile proprietario ($q' = w\/"cap"_b$, colonna $w - q' dot "cap"_b$) e se ne legge l'ultima riga, oppure si scrivono zeri se $b = 0$. Le righe interne applicano il solito massimo fra il caso "skip" (riga precedente, stessa cella: lettura sempre interna) e il caso "take" (cella $w - w_i$): se $w - w_i >= "lo"$ la lettura resta nel tile, altrimenti va nel tile a sinistra della stessa riga.

La funzione di alto livello `knapsackdpdag` incolla le fasi: costruzione del grafo, risoluzione, ricostruzione della soluzione e al termine registra in `DAGStats` il numero di tile ($n_b dot n_q$), di archi e di livelli, insieme alla larghezza delle frontiere (tile per livello, riassunta con media, mediana, minimo e massimo): sono le statistiche riportate in @tab:dag-frontiers nella sezione dei risultati.

== Knapsack DP DAG: Risultati

=== Esecuzione dell'esperimento

Le misure condividono l'ambiente del capitolo precedente (nodo ARM Neoverse-N1 del cluster mercurio, `kpbench` compilato con GCC 14.2.0 in preset Release, misure `perf stat` ). Questo consente un confronto diretto fra le due versioni: tempi e contatori provengono dalla stessa campagna.

=== Scalabilità e crossover

#figure(
  image("../images/perf_kpdpdag_speedup.png", width: 100%),
  caption: [Speedup del KnapsackDPDAG a $n = 50$ per famiglia di peso e capacity.],
) <fig:dag-speedup-cap>

A $n = 50$ lo sweep sulla capacity mostra uno speedup a muro che si ferma nell'ordine di $2.2 - 2.4 times$ qualunque siano famiglia di peso e capacity, e l'*insn ratio* a 32 thread sale a $4.4 times$: il run parallelo esegue 4,4 volte le istruzioni del seriale, quindi circa il 77% delle sue istruzioni ($1 - 1\/4.4$) è lavoro in più. Il solver a livelli non usa code né operazioni atomiche: quel lavoro in più è attesa attiva dei thread sulla barriera implicita di fine `parallel for`, a ogni livello. Il collo non dipende dunque dalla taglia del problema ma dal rapporto fra lavoro per tile e costo di sincronizzazione su frontiere strette.

#figure(
  image("../images/perf_kpdpdag_speedup_vs_n.png", width: 100%),
  caption: [Speedup del KnapsackDPDAG al variare di thread e numero di oggetti (capacity $10^6$, granularità tile migliore per punto, mediana su 2 repliche).],
) <fig:dag-speedup>

Tenendo invece fissata la capacity e variando il numero di oggetti il quadro cambia radicalmente. La campagna di @fig:dag-speedup ripete lo sweep sui thread per $n in {50, 250, 1000, 5000}$ a capacity $10^6$, scegliendo per ogni punto la migliore fra quattro granularità ($"item"_b in {5, 25}$, $"cap"_b in {10^3, 10^4}$): a $n = 50$ lo speedup si ferma a $approx 2.7 times$, cresce già a $approx 6.1 times$ con $n = 250$ e raggiunge $approx 8.1 - 8.4 times$ per $n >= 1000$. La griglia include anche granularità con frontiere più larghe ($"cap"_b = 10^3$: $n_q = 1001$ blocchi di capacità invece di $101$, picco di frontiera $approx 1000$ tile invece di $approx 101$): a parità di $"item"_b$ aggiungono parallelismo esposto ma non vincono mai nel regime parallelo; a $n = 5000$ e 32 thread $(5, 10^3)$ impiega $approx 5.1 s$ contro $approx 1.9 s$ di $(5, 10^4)$. La ragione è che la frontiera stretta copre già i 32 thread ($approx 101$ tile di picco): la larghezza in più non espone lavoro aggiuntivo, mentre frammenta lo stesso calcolo in $10 times$ più tile, ognuno con $10 times$ meno lavoro utile e gli stessi costi fissi di schedulazione e copia della riga di confine, e raddoppia i livelli ($2000$ contro $1100$).

#figure(
  table(
    columns: 6,
    [$n$], [$T_"dag"(1)$], [$T_"plain"(1)$], [vantaggio seriale], [$S_"wall"$ DAG $p=32$], [insn ratio DAG $p=32$],
    [50], [0.16 s], [0.64 s], [$4.0 times$], [$2.4 times$], [$4.4 times$],
    [1000], [3.41 s], [23.1 s], [$6.8 times$], [$7.1 times$], [$1.35 times$],
    [5000], [15.9 s], [150 s], [$9.4 times$], [$9.1 times$], [$1.25 times$],
  ),
  caption: [Crossover dello scaling: stesso problema, stessa capacity, numero di oggetti crescente (2 repliche, spread $<= 1.03$).],
) <tab:dag-crossover>



Il limite di scala ha anche una spiegazione strutturale: il parallelismo effettivamente esposto dal DAG dipende dalla distribuzione delle frontiere sui livelli. La geometria del DAG è governata da due parametri di tiling: `item_block` ($"item"_b$ nelle tabelle) è il numero di oggetti, cioè di righe della DP, accorpati in un singolo tile, `cap_block` ($"cap"_b$) è il numero di colonne di capacità per tile; la griglia risulta così composta da $n_b = ceil(n\/"item"_b)$ blocchi di item e $n_q = ceil((C+1)\/"cap"_b)$ blocchi di capacità, per un totale di $n_b dot n_q$ tile, ciascuno responsabile del calcolo di un sottoblocco $"item"_b times "cap"_b$ della tabella. Blocchi grandi riducono il numero di tile e con esso l'overhead di schedulazione e l'indirezione cross-tile, ma stringono le frontiere (meno tile indipendenti per livello); blocchi piccoli fanno il contrario, esponendo più parallelismo a costo di più sincronizzazioni. Il solver riporta le statistiche delle frontiere (larghezza = numero di tile per livello) per ogni esecuzione; la tabella seguente le riassume per alcune configurazioni,

#figure(
  table(
    columns: 8,
    [$n$], [$"item"_b$], [$"cap"_b$], [tile], [livelli], [$"min"$], [$"mediana"$], [$"media"\/"max"$],
    [50], [5], [$10^2$], [100010], [10010], [1], [10.0], [9.99 \/ 10],
    [50], [5], [$10^3$], [10010], [1010], [1], [10.0], [9.91 \/ 10],
    [50], [5], [$10^4$], [1010], [110], [1], [9.9], [9.18 \/ 10],
    [50], [10], [$10^2$], [50005], [10005], [1], [5.0], [5.00 \/ 5],
    [50], [10], [$10^3$], [5005], [1005], [1], [5.0], [4.98 \/ 5],
    [50], [10], [$10^4$], [505], [105], [1], [5.0], [4.81 \/ 5],
    [50], [25], [$10^2$], [20002], [10002], [1], [2.0], [2.00 \/ 2],
    [50], [25], [$10^3$], [2002], [1002], [1], [2.0], [2.00 \/ 2],
    [50], [25], [$10^4$], [202], [102], [1], [2.0], [1.98 \/ 2],
    [5000], [5], [$10^2$], [10001000], [11000], [1], [993.2], [909.2 \/ 1000],
    [5000], [5], [$10^3$], [1001000], [2000], [1], [478.8], [500.5 \/ 1000],
    [5000], [5], [$10^4$], [101000], [1100], [1], [100.3], [91.8 \/ 101],
    [5000], [10], [$10^2$], [5000500], [10500], [1], [499.1], [476.2 \/ 500],
    [5000], [10], [$10^3$], [500500], [1500], [1], [368.4], [333.7 \/ 500],
    [5000], [10], [$10^4$], [50500], [600], [1], [97.9], [84.2 \/ 101],
    [5000], [25], [$10^2$], [2000200], [10200], [1], [199.9], [196.1 \/ 200],
    [5000], [25], [$10^3$], [200200], [1200], [1], [193.9], [166.8 \/ 200],
    [5000], [25], [$10^4$], [20200], [300], [1], [74.6], [67.3 \/ 101],
  ),
  caption: [Statistiche delle frontiere del DAG (larghezza = tile per livello)],
) <tab:dag-frontiers>

La struttura del grafo rende la distribuzione deterministica: oltre all'arco col tile sopra $(b-1, q)$, il range di dipendenza shiftato di $w_i$ crea sistematicamente un arco col tile alla sua sinistra $(b, q-1)$, quindi il livello di un tile è $"level"(b,q) = b+q$ e le frontiere sono le anti-diagonali della griglia di tile. Ne segue che i livelli sono $n_b + n_q - 1$, il picco di larghezza è $min(n_b, n_q)$ e la media $n_b n_q\/(n_b+n_q-1)$: tutti i valori misurati in @tab:dag-frontiers coincidono con queste formule. La relazione presuppone pesi $w_i >= 1$ e $"maxWeight" <= "cap"_b$: con pesi nulli, o pesi tali da scavalcare più blocchi di capacità, la struttura delle dipendenze sarebbe diversa, pur restando valido l'ordinamento topologico row-major. La conseguenza è che a $n = 50$ il picco di parallelismo esposto è solo $n_b = ceil(n\/"item"_b)$, cioè fra 2 e 10 tile per livello a seconda della granularità: con 32 esecutori la maggior parte resta ferma a ogni frontiera, e questo è il vero limite di scaling della versione DAG a pochi oggetti. Il valore cresce con $n$ (a $n = 1000$ il picco arriva a 101 tile, oltre il numero di thread) e spiega perché le curve di speedup di @fig:dag-speedup si innalzino col numero di oggetti, in accordo con i valori di @tab:dag-crossover, così come la regola empirica della granularità (più thread richiedono `item_block` più piccolo, cioè più livelli e frontiere più larghe, fino alla saturazione a $n_q$ discussa sopra).

=== Un limite superiore allo speedup del solver a livelli <sec:dag-bound>

Le formule delle frontiere danno direttamente un limite allo speedup ottenibile dal solver a livelli, indipendente dalla macchina. Se i tile hanno lo stesso costo $tau$, il livello $L$ con $|F_L|$ tile richiede $ceil(|F_L| \/ p) dot tau$ con $p$ thread, per cui

$
  S_"max"(p) = (n_b dot n_q) / (sum_L ceil(|F_L| \/ p)), quad lim_(p -> infinity) S_"max"(p) = (n_b n_q) / (n_b + n_q - 1)
$

cioè il rapporto fra lavoro e profondità (*work/span*) del grafo dei tile. A $n = 50$ con $"item"_b = 10$ e $"cap"_b = 10^4$, $C = 10^6$ ($n_b = 5$, $n_q = 101$, ogni frontiera ha al più 5 tile) il limite vale $505 \/ 105 approx 4.8$ per ogni $p >= 5$: lo speedup misurato di $approx 2.4$ è circa la metà del massimo teorico. Il confronto fra la curva misurata e $S_"max"(p)$ separa ciò che manca per struttura del grafo (frontiere strette) da ciò che si perde per sincronizzazione e banda. #todo[grafico di $S_"max"(p)$ sovrapposto agli speedup misurati con `run_timing.sbatch`.]

=== Variante a wavefront esplicito, senza grafo

Con pesi $w_i >= 1$ il livello di un tile è $b + q$ (@tab:dag-frontiers), e la visita per anti-diagonali è comunque valida per *qualsiasi* insieme di pesi: la riga $i$ del tile $(b, q)$ legge la riga $i-1$ nelle colonne $w$ e $w - w_i <= w$, cioè nel tile $(b, q')$ o, per la prima riga del blocco, $(b-1, q')$ con $q' <= q$, e tutti questi stanno su anti-diagonali precedenti o sul tile stesso. Quando invece $"maxWeight" > "cap"_b$ alcuni tile non dipendono dal vicino di sinistra e il DAG può assegnare livelli più bassi di $b + q$, cioè frontiere più larghe: è il caso in cui il grafo esplicito può esporre più parallelismo del wavefront, e lo sweep su maxWeight e $"cap"_b$ di `run_timing.sbatch` serve a misurarlo. La variante `knapsackdptiled` usa questa visita direttamente sulla tabella row-major della versione classica: stessi tile e stesso parallelismo del DAG, ma senza costruzione del grafo, senza un'allocazione per tile e senza copia della riga di confine. Serve a un confronto per esclusione (*ablation*):

- classica row-major contro `knapsackdptiled` a un thread misura il solo effetto della visita a tile sulla località;
- `knapsackdptiled` contro `knapsackdpdag` allo stesso numero di thread misura il costo del grafo e dei blocchi separati.

#todo[tabella dell'ablation dalle misure di `run_timing.sbatch`.]

=== Lo scheduler topologico alternativo

Oltre al solver a livelli, `solve_dag_topo` (attivato da `KNAPSACK_DAG_TOPO=1`) esegue i tile appena il loro grado d'ingresso scende a zero, con una coda condivisa. Nella prima versione la coda pubblicava gli slot con un unico contatore `filled = slot + 1`: con due produttori concorrenti il secondo poteva rendere visibile lo slot del primo prima che fosse scritto, e un consumatore leggeva un valore non inizializzato (un tile calcolato due volte e uno mai, con stallo o risultato errato). Ora ogni slot è pubblicato da solo. Le misure di questo capitolo non usano lo scheduler topologico.

=== L'effetto del tiling sulla località

#figure(
  image("../images/perf_vs_algorithms.png", width: 100%),
  caption: [Confronto KnapsackDP classico e KnapsackDPDAG a capacity $10^6$: IPC, cicli per cella e LLC MPKI.],
) <fig:dag-vs-classic>

#nota(titolo: "Confronto da rifare")[Questa sezione e la successiva confrontano il DAG con la DP classica *column-major* (@sec:kpdp-layout): la differenza di IPC, di miss e di traffico DRAM misura soprattutto il layout della tabella classica, non il tiling. Con la tabella row-major la classica è più veloce del DAG già in seriale sulla macchina della verifica preliminare. Le conclusioni vanno riscritte sull'ablation classica / `knapsackdptiled` / `knapsackdpdag`.]

Il guadagno principale del tiling è sulla gerarchia di memoria. In seriale a capacity $10^6$ la versione DAG raggiunge un IPC di $2.2 - 2.3$ contro $approx 0.64$ del classico, con $approx 4$ miss LLC per 1000 istruzioni contro $approx 97$: le miss totali crollano da $121 times 10^6$ a $5 times 10^6$ ($24 times$ in meno) a parità di istruzioni eseguite ($approx 1.2 times 10^9$ per entrambe). Il costo per cella scende da $approx 38 - 40$ a $approx 11$ cicli: il tile mantiene il proprio working set in cache e la riga di confine del tile superiore resta residente.

Il vantaggio seriale è $approx 4 times$ (0.16 s contro 0.64 s) e *cresce* con il numero di oggetti: il costo per cella del classico degrada da 12.6 a 30.0 ns/cella passando da $n = 50$ a $n = 5000$ (la tabella, da 4 a 20 GB, peggiora cache e TLB), mentre quello del DAG resta piatto a $approx 3.2 - 3.4$ ns/cella: a $n = 5000$ il vantaggio seriale arriva a $9.4 times$.

=== La granularità dei tile

#figure(
  image("../images/perf_kpdpdag_block_grid.png", width: 100%),
  caption: [Effetto di item_block e cap_block su tempo, IPC e LLC MPKI (maxWeight=10, $n = 5000$, capacity $10^6$, 32 thread).],
) <fig:dag-block-grid>

La scelta della granularità governa il compromesso fra parallelismo esposto e overhead di schedulazione. Con pochi oggetti ($n = 50$) vince la granularità grossa a thread bassi: `item_block` = 25 è ottimo a 1--2 thread, 15 a 4, 10 a 8 e 5 oltre i 16 thread (più thread richiedono più tile indipendenti). @fig:dag-block-grid ripete il confronto alla taglia opposta, $n = 5000$, capacity $10^6$ e 32 thread: vince ($"item"_b = 10$, $"cap"_b = 10^4$) con $1.75 s$; `item_block` = 25 realizza l'IPC più alto ($1.67$ contro $1.43$, meno sincronizzazioni per item) ma lascia frontiere mediamente più corte ($approx 67$ contro $84$ tile) e perde in tempo ($2.07 s$); `cap_block` = $10^3$ moltiplica per dieci il numero di tile senza alcun guadagno di località (l'LLC MPKI non scende mai sotto $3.2$) e costa fino a $approx 2.2 times$ in tempo. Il `cap_block` grande ($10^4$) resta quindi preferibile a capacity $10^6$ con molti thread, così come per maxWeight $= 1000$.



== Confronto tra la versione classica e la versione DAG

#figure(
  image("../images/perf_big_crossover.png", width: 100%),
  caption: [Speedup del DAG sul classico $T_"plain"\/T_"dag"$ in tempo assoluto a capacity $10^6$, per numero di oggetti crescente: sempre sopra 1; il DAG è più veloce a ogni thread count, con vantaggio massimo in seriale.],
) <fig:kpdp-perf-crossover>

A capacity $10^6$ la versione DAG è più veloce del classico a tutti i thread count, come mostra @fig:kpdp-perf-crossover: lo speedup del DAG sul classico passa da $4.0 times$ a $p = 1$ a $1.28 times$ a $p = 32$ per $n = 50$, e da $9.4 times$ a $5.8 times$ per $n = 5000$. Il vantaggio è massimo in seriale, dove coincide con il rapporto fra i tempi seriali, e si comprime al crescere dei thread, perché il limite di sincronizzazione del DAG appiattisce la sua curva mentre il classico continua a scalare. La ragione è strutturale: il classico è memory-bound per costruzione ($approx 97$ LLC MPKI, hit $approx 0%$, IPC $approx 0.6$), e aggiungere thread non cura il problema perché i thread extra aggiungono contesa, non località ($K_c$ che cresce di $4 times$, insn ratio $+33%$, entrambi a capacity $10^6$). Il tiling attacca la causa e riduce il traffico DRAM di $approx 17 times$ (33 GB contro 569 GB di letture stimate a $n = 5000$, $p = 32$).

