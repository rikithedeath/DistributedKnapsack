#import "functions/preamble.typ": *
#show: style-doc

#align(center)[
  #text(size: 17pt, weight: "bold")[Knapsack parallelo: implementazione, comparazione e analisi]
  #v(0.4em)
  #text(size: 12pt)[Matteo Ielacqua]
  #v(1em)
]

#outline(depth: 2)

#heading(numbering: none)[Sommario]
Il documento confronta quattro organizzazioni parallele in memoria condivisa della soluzione per programmazione dinamica dello zaino 0/1, costruite sullo stesso kernel interno: la versione classica, che parallelizza la singola riga della tabella; la stessa con due righe e una tabella di bit per la ricostruzione; una versione a tile eseguita per anti-diagonali; e una versione a tile in cui un grafo aciclico delle dipendenze fra tile (DAG) definisce le frontiere che si possono eseguire in parallelo. Per il DAG si ricava un limite superiore allo speedup dalla struttura delle frontiere; un'analisi per esclusione separa l'effetto della località, del parallelismo e del costo del grafo. In coda, come extra, la versione CUDA dell'algoritmo classico.

#nota(titolo: "Stato delle misure")[La prima campagna ha misurato la DP classica con la tabella memorizzata per colonne, che rallenta la versione classica di circa un ordine di grandezza (@sec:kpdp-layout). Le sezioni che la usano sono segnalate e vanno aggiornate con le campagne descritte in @sec:metodologia; i valori ancora da inserire sono marcati in rosso.]

= Introduzione
#include "chapters/introduzione.typ"

#include "chapters/metodologia.typ"

#include "chapters/knapsack_dp_classic.typ"

#include "chapters/knapsack_dp_dag.typ"

#include "chapters/knapsack_dp_gpu.typ"

#include "chapters/conclusioni.typ"

#bibliography("biblio.bib", title: "Bibliografia", style: "ieee")

#include "chapters/appendice.typ"
