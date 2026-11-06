#import "../functions/preamble.typ": *

= Limiti dello studio

- *una sola architettura*: tutte le misure provengono da un nodo Arm Neoverse-N1. I rapporti fra le varianti dipendono dalla gerarchia di memoria e possono cambiare su un'altra CPU;
- *istanze sintetiche*: pesi e valori uniformi, e capacità spesso superiori alla somma dei pesi (@sec:metodologia);
- *memoria*: le varianti con la tabella completa occupano $4 (n+1)(C+1)$ byte (20 GB a $n = 5000$, $C = 10^6$), il che limita le istanze misurabili; la variante a bit riduce questo costo di 32 volte;
- *misure della prima campagna*: tempi e contatori della DP classica, e i confronti che la usano, sono stati raccolti con la tabella column-major (@sec:kpdp-layout) e vanno sostituiti dalle nuove campagne.

= Conclusioni

#todo[da riscrivere sui risultati di `run_timing.sbatch` e delle campagne `perf` aggiornate.]

Indipendentemente dai numeri, il lavoro stabilisce tre risultati:

- la DP dello zaino si parallelizza in modo naturale sulla riga (modello CREW, profondità $O(n)$); il suo limite pratico è la banda di memoria, da misurare con il roofline e non da dedurre dai soli contatori;
- con pesi $w_i >= 1$ e $"maxWeight" <= "cap"_b$ il grafo dei tile ha livelli $b + q$: il parallelismo esposto è $min(n_b, n_q)$ e lo speedup del solver a livelli è limitato da $n_b n_q \/ (n_b + n_q - 1)$, che a pochi oggetti è piccolo qualunque sia il numero di thread;
- l'ordine di memorizzazione della tabella pesa più di qualunque scelta di parallelizzazione: la stessa DP classica cambia di un ordine di grandezza passando da column-major a row-major. Un confronto fra algoritmi è significativo solo se le varianti condividono layout e kernel interno, che è il motivo dell'ablation classica / `knapsackdptiled` / `knapsackdpdag`.
