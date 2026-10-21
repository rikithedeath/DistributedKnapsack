#import "../functions/preamble.typ": *


Prima dell'analisi dei risultati, bisogna tenere a mente alcune premesse. La prima, la più importante, è che l'algoritmo COPA proposto non può funzionare con più di 50 oggetti, questo vincolo comunque non è particolarmente oneroso per l'analisi, visto che comunque nella versione DP dell'algoritmo si può parallelizzare solo sul ciclo più interno, la cui lunghezza dipende dalla capacità. La seconda è che nel caso di un algoritmo MPI, un implementazione puramente distribuita è sicuramente molto più inefficiente di una ibrida, che però richiederebbe degli accorgimenti ulteriori per essere implementata.

== Shared Memory

i due algoritmi in shared memory hanno efficacia profondamente diversa.

=== Knapsack DP

#figure(
  image("../images/t_c_p_shared.png", width: 100%),
  caption: [Confronto dei tempi di esecuzione per gli algoritmi shared memory al variare del numero di thread.],
) <fig:tcpshared>

Dal grafico soprastante si osserva innanzitutto che, superata una certa capacità, i tempi di esecuzione dell'algoritmo DP crescono rapidamente, mentre quelli del COPA rimangono sostanzialmente costanti: questo è dovuto al fatto che la complessità del DP dipende dal prodotto $N times C$ (con $C$ la capacità dello zaino), mentre il COPA dipende solo dal numero di elementi $N$, generando $2^(N\/2)$ sottoinsiemi indipendentemente dalla capacità.

#figure(
  image("../images/KnapsackDP_Speedup.png", width: 100%),
  caption: [Speedup dell'algoritmo KPDP in shared memory],
) <fig:kpdpshared>

Lo speedup nel caso della versione in dynamic programming come ci si potrebbe aspettare cresce un pò con l'aumentare massiccio dei threads, questo diviene particolarmente vero quando la capacità è molto alta, sopratutto perchè a quel punto il ciclo interno sarà quello con più lavoro da eseguire. Questo comportamento smette di verificarsi quando non ci sono abbastanza elementi da poter giustificare l'instanziazione di tutti quei thread e quindi si perde di efficacia, ciò è verificabile meglio nel plot sottostante

#figure(
  image("../images/KnapsackDP_speedupplot.png", width: 100%),
  caption: [Speedup plot dell'algoritmo KP in DP],
) <fig:kpdpsharedplot>

In questo grafico sono riportati diversi speedup a seconda della capacità, si vede subito comunque che l'algoritmo scala bene solo con i primi threads e solo quando la capacità supera una certa soglia. Diventa ancora più evidente se si osserva il grafico dell'efficienza

#figure(
  image("../images/KnapsackDP_Efficiency.png", width: 100%),
  caption: [Efficienza dell'algoritmo KP in DP],
) <fig:kpdpsharedefficiency>

=== Knapsack COPA

Nell'algoritmo COPA si può osservare invece che lo speedup cresce molto rapidamente all'aumentare dei processori ma decresce subito superata una certa soglia

#figure(
  image("../images/KnapsackCOPA_Speedup.png", width: 100%),
  caption: [Speedup dell'algoritmo KP COPA in shared memory],
) <fig:kpcopaspeedup>

#figure(
  image("../images/KnapsackCOPA_speedupplot.png", width: 100%),
  caption: [Speedup dell'algoritmo KP COPA in shared memory],
) <fig:kpcopaplot>

Non bisogna sorprendersi della discesa repentina dello speedup all'aumentare dei processori, l'algoritmo presenta infatti una peculiarità: ovvero che potrebbe scalare meglio con un numero di elementi più alto, sfortunatamente il numero di soluzioni generate durante lo step di creazione dei subsets lo rende proibitivo a livello di memoria richiesta, visto che lo spazio occupato è $2^n$ elementi. Le linee a capacità bassa che segnano invece uno speedup di 1.4 sono perlopiù falsi positivi dovuti alla chiusura praticamente immediata del job vista l'assenza di coppie valutabili durante la fase di pruning. Ovviamente questo comportamento si riflette sull'efficienza che decresce repentinamente e sui costi che salgono rapidamente sopratutto oltre i 20 processori.

#figure(
  image("../images/KnapsackCOPA_Cost.png", width: 100%),
  caption: [Costo dell'algoritmo KP COPA in shared memory],
) <fig:kpcopacost>

#figure(
  image("../images/KnapsackCOPA_Efficiency.png", width: 100%),
  caption: [Efficienza dell'algoritmo KP COPA in shared memory],
) <fig:kpcopaefficiency>

=== Knapsack DP vs COPA

Mettendo a confronto i due algoritmi si può osservare che, utilizzando l'algoritmo COPA nelle giuste configurazioni si possono ottenere benefici considerevoli in termini di speedup a un costo contenuto in termini di processori impiegati.

#figure(
  image("../images/knapsackdpvscopashared.png", width: 100%),
  caption: [Speedup dell'algoritmo KP COPA vs DP],
) <fig:kpcopavsdp>

Come si può osservare dal grafico, l'algoritmo in dynamic programming vince sostanzialmente in tutte quelle configurazioni in cui la capacità è più piccola di una certa soglia, sotto il $10^7$, mentre invece per configurazioni in cui la capacità è molto grande COPA è molto più veloce. Questo conferma quanto scritto nell'articolo in cui si presenta proprio questa soluzione al problema dello zaino, va detto comunque che la limitazione d'uso a cui si è sottoposti è importante, visto che non è possibile calcolare soluzioni per insiemi più grandi di 50 elementi.

== Distributed Memory

Entrambi gli algoritmi, come si può osservare nel grafico, funzionano molto male quando si utilizzano in memoria distribuita.

#figure(
  image("../images/knapsackdp_mpi_mpi.png", width: 100%),
  caption: [Speedup degli algoritmi in distributed memory],
) <fig:kpcopadpmpi>

Nel caso dell'algoritmo in dynamic programming, la motivazione è semplice da individuare: il broadcast della riga ad ogni iterazione comporta un costo alto in termini di tempi di comunicazione. Supponendo di usare un intero a 32 bit, quindi 4 byte, per rappresentare il vettore di pesi e il vettore dei profitti, se si suppone una lunghezza di $10^8$ elementi bisogna trasferire circa 381 Mb ogni volta solo per il broadcast e altrettanti per eseguire il gathering e lo scattering del prossimo vettore da calcolare, rendendo questo algoritmo altamente memory bound. La versione COPA invece non presenta questo problema durante il calcolo, purtroppo durante la fase di generazione dei subsets rende il processo memory bound almeno inizialmente, visto che per ogni elemento è necessario scambiarsi tutta la lista di subsets generati, supponendo di usare 3 interi a 32 bit per rappresentare profitto, peso e indice del blocco servono $2^(n\/2) * 4 * 3 * 2$ al passo finale per entrambe le liste, per cui in totale bisogna trasferire $sum_(i = 1)^n 2^(i\/2) * 4 * 3 * 2$, misura che diventa considerevolmente grande una volta che il numero di elementi diventa abbastanza grande, rendendo di fatto l'algoritmo nuovamente memory bound.

=== Possibili miglioramenti

L'approccio a memoria condivisa utilizzato è molto vicino a quello ottimale, usando un dispositivo come una GPU sarebbero possibili accorgimenti ulteriori per migliorare l'efficienza: l'approccio COPA è infatti stato pensato proprio per un approccio GPU oriented, per esempio nella fase di generazione dei subset sarebbe possibile, avendo abbastanza core a disposizione, aggiungere in pochi step l'elemento corrente a tutti gli elementi della lista shifted. Con opportuni accorgimenti potrebbe essere possibile utilizzare
direttamente la versione OpenMP eseguendo offloading a un dispositivo GPU, oppure si potrebbe usare OpenACC per ottenere il medesimo risultato.

Nell'approccio a memoria distribuita le modifiche possibili sono invece più sostanziose e porterebbero sicuramente a incrementi non banali in termini di prestazioni, prima di tutto l'implementazione di un approccio puramente a memoria distribuita, come si è osservato, porta inevitabilmente al dilatamento dei tempi a causa degli elevati tempi di trasmissione, inoltre nessuno di questi algoritmi è stato pensato per funzionare con un approccio a memoria distribuita ma entrambi per la memoria condivisa. Tuttavia sono comunque possibili dei miglioramenti, in entrambe le versioni invece di adottare una strategia puramente a memoria distribuita si può tentare un approccio ibrido, in cui ai task vengono date più cpu per effettuare operazioni in memoria condivisa, consentendo così di distribuire in modo più grossolano i task e evitando ritardi dovuti alle distribuzioni granulari che avvengono adesso, come i broadcast o gli all_gather utilizzati per condividere quei dati che costituiscono una dipendenza per i passi successivi dell'algoritmo in questione.

== Conclusioni

Lo scopo primario del progetto era comprendere a pieno la difficoltà di implementazione dell'algoritmo e il beneficio ottenibile. L'approccio shared memory ha mostrato di essere promettente, sopratutto per quanto riguarda l'approccio COPA. Tuttavia proprio quell'approccio è assolutamente inutilizzabile nel mondo embedded, dove la memoria è una risorsa molto scarsa e quindi determinati approcci non sono possibili. Comunque se si utilizzasse invece per implementare un ottimizzazione da terra da poi usare in orbita, potrebbe rivelarsi una scelta molto promettende per tutti quei problemi che hanno un vincolo di capacità molto elevato con valori di pesi sugli oggetti estremamente variabili tra di loro.
