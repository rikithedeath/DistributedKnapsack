#import "functions/preamble.typ": *
#show: style-doc

#align(center)[
  #text(size: 17pt, weight: "bold")[Knapsack parallelo: implementazione, comparazione e analisi]
  #v(0.4em)
  #text(size: 12pt)[Matteo Ielacqua]
  #v(1em)
]

= Sommario
Il documento riassume tre approcci all'implementazione parallela dell'algoritmo dello zaino nella sua versione dynamic programming. La prima parte del documento contiene l'algoritmo nella sua versione classica, insegnata nei corsi di programmazione , che lo risolve sulla frontiera di esecuzione naturale, cioè la singola linea della tabella; dopo un iniziale spiegazione dell'algoritmo segue la presentazione dei risultati. Nella seconda parte viene affrontato lo stesso algoritmo ma con un approccio diverso, per costruire le frontiere viene infatti utilizzato un Directed Acyclic Graph costituito da tile presi dalla tabella, questi tile vengono organizzati poi in frontiere denominate livelli di blocchi che possono essere eseguiti parallelamente. L'ultima parte, presentata come extra, affronta l'integrazione dell'algoritmo originale in versione CUDA per essere usato su schede grafiche NVIDIA.

#include "chapters/knapsack_dp_classic.typ"

#include "chapters/knapsack_dp_dag.typ"

#include "chapters/knapsack_dp_gpu.typ"
