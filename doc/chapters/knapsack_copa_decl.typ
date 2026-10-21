
#import "@preview/algorithmic:1.0.7"
#import algorithmic: Line, algorithm-figure, style-algorithm
#import "@preview/codelst:2.0.2": sourcecode
#import "../functions/preamble.typ": *

=== COPA: Cost Optimal Parallel Algorithm

Un interessante modello di esecuzione è denominato COPA @knapsackCopaSolution e sviluppa un algoritmo parallelo cost optimal per
risolvere il problema sia su CPU con OpenMP che su GPU con CUDA, l'idea dietro quest'algoritmo è la seguente: 1) Si prende
l'array di oggetti $V$ e lo si divide in 2 insiemi disgiunti. 2) Per il primo insieme si generano tutte le $N = 2^(n\/2)$ possibili
combinazioni $a_i$, si calcola di ogni sottoinsieme $a_i$ peso e profitto $w_i, p_i$ e si aggiunge questa tripletta in una lista
in ordine crescente. 3) Si esegue lo stesso con la seconda lista per ogni sottoinsieme $b_i$ e le triplette si mettono in una lista
di ordine non crescente. In questo modo si è generata la lista A, per generare la lista B si riproducono gli step precedenti
semplicemente inserendo le triplette in una lista di ordine non crescente. A questo punto si può passare alla fase di ricerca
che prenderà in input le due liste ordinate.

#algorithm-figure(
  "Algoritmo di ricerca",
  {
    import algorithmic: *
    reqline[Due liste ordinate $A$ e $B$]
    ensureline[Soluzione finale $"Bestvalue"$ e $X$]
    Line[Inizializza $"MaxB"_N <- b_N dot "p"$, $"L"_N <- N$, $"Bestvalue" <- 0$, $"X"_1 <- (0, 0)$]
    For($"i" <- N - 1 space "downto" space 1$, {
      If(
        $b_i dot "p" > "MaxB"_(i+1)$,
        {
          states[$"MaxB"_i <- b_i dot "p" space "and" space "L"_i <- i$]
        },
        {
          states[$"MaxB"_i <- "MaxB"_(i+1) space "and" space "L"_i <- "L"_(i+1)$]
        },
      )
    })
    states[$i <- 1 space "and" space j <- 1$]
    While($i <= N space "and" space j <= N$, {
      If($a_i dot "w" + b_j dot "w" > c$, {
        states[$j <- j + 1 space "and" space "continue"$]
      })
      If($a_i dot "p" + "MaxB"_j > "Bestvalue"$, {
        states[$"Bestvalue" <- a_i dot "p" + "MaxB"_j space "and" space "X"_1 <- (a_i, b_(L_j))$]
      })
      Assign[$i$][$i + 1$]
    })
    Line[Converti le due componenti decimali di $X_1$ in due numeri binari: $X_1 <- ("binary"_1, "binary"_2)$]
    LineComment(Assign[$X$][Call["strcat"][$"binary"_1$, $"binary"_2$]], [Concatena i due binari])
  },
) <alg:cap>


Questo modello di esecuzione si può vedere subito che si presta bene ad essere parallelizzato, sia nella parte di generazione
delle liste sia in quella poi di ricerca dell'ottimo.



L'algoritmo parallelo è stato pensato per essere risolto su un modello EREW (Exclusive Read, Exclusive Write) PRAM con shared memory e assume $k$ processori accessibili.
L'algoritmo sfrutta prima un'algoritmo di generazione parallela delle liste A e B

#algorithm-figure(
  "Algoritmo parallelo di generazione",
  {
    import algorithmic: *
    reqline[Elementi $(v_i, v_i dot "w", v_i dot "p")$, $i = 1, 2, dots, n/2$]
    ensureline[Lista ordinata $A_(n/2)$ (i.e. $A$)]
    Line[Inizializza $A_0 <- [(0, 0, 0)]$ e $"IC" <- 1$]
    For($i <- 0 space "to" space n/2 - 1$, {
      Assign[$v_(i+1)$][$"IC"$]
      For([#strong[all] $P_j$, $1 <= j <= k$, #strong[in parallel]], {
        Line[Genera nuova lista $A'_i$ aggiungendo la tripletta $(v_(i+1), v_(i+1) dot "w", v_(i+1) dot "p")$ a tutti gli elementi della lista ordinata $A_i$]
        Line[Chiama l'algoritmo di merge parallelo per unire $A_i$ e $A'_i$ in ordine non decrescente di peso, producendo la lista ordinata $A_(i+1)$]
      })
      Assign[$"IC"$][$"IC" + 1$]
    })
  },

) <alg:parallel_gen>

l'algoritmo di merging parallelo si articola nel modo seguente: prende in input due vettori ordinati $U = (u_1, u_2, dots, u_m)$ e $V = (v_1, v_2, dots, v_m)$ e $k$ processori $P_1, P_2, dots, P_k$, con $1 <= k <= 2m$ potenza di 2, e produce un vettore ordinato di lunghezza $2m$ ottenuto dall'unione di $U$ e $V$.

+ #text(weight: "bold")[Step 1:] I $k$ processori partizionano $U$ e $V$ in parallelo, ciascuno in $k$ sottovettori (che possono essere vuoti) $(U_1, U_2, dots, U_k)$ e $(V_1, V_2, dots, V_k)$ tali che: $|U_i| + |V_i| = 2m\/k$ per $1 <= i <= k$, e i pesi di tutti gli elementi in $U_i$ e $V_i$ sono minori dei pesi di tutti gli elementi in $U_(i+1)$ e $V_(i+1)$, per $1 <= i <= k - 1$.
+ #text(weight: "bold")[Step 2:] Il processore $P_i$ esegue il merge sequenziale di $U_i$ e $V_i$, per $1 <= i <= k$, in parallelo con gli altri processori.

A questo punto si può passare a dividere equamente tra i processori disponibili i blocchi ottenuti e calcolare per ognuno di questi blocchi gli elementi di profitto massimo.

#algorithm-figure(
  "ALgoritmo di divisione e ricerca del massimo profitto per blocco",
  {
    import algorithmic: *
    reqline[Due liste ordinate $A$ e $B$]
    ensureline[$"MaxA"$ e $"MaxB"$]
    Line[Dividi sia $A$ che $B$ uniformemente in $k$ blocchi]
    For([#strong[all] $P_i$, $1 <= i <= k$, #strong[in parallel]], {
      states[$"MaxA"_i <- a_(i,1) dot "p"$, $"MaxB"_i <- b_(i,1) dot "p"$]
      For($j <- 2 space "to" space e$, {
        If($a_(i,j) dot "p" > "MaxA"_i$, {
          Assign[$"MaxA"_i$][$a_(i,j) dot "p"$]
        })
        If($b_(i,j) dot "p" > "MaxB"_i$, {
          Assign[$"MaxB"_i$][$b_(i,j) dot "p"$]
        })
      })
    })
  },

) <alg:parallel_max>

Ora lo spazio delle soluzioni è davvero molto grande, per questo motivo si ricorre a un algoritmo di potatura dell'insieme delle soluzioni

#algorithm-figure(
  "Algoritmo di pruning parallelo",
  {
    import algorithmic: *
    reqline[Due liste di blocchi $A = {A_1, A_2, dots, A_k}$ e $B = {B_1, B_2, dots, B_k}$, capacità $c$]
    ensureline[Insieme di coppie di blocchi non potate $"remaining"$]
    Line[Inizializza $"remaining" <- emptyset$]
    For([#strong[all] $P_i$, $1 <= i <= k$, #strong[in parallel]], {
      Line[Inizializza $"local_results" <- emptyset$]
      For($j <- i space "to" space k + i - 1$, {
        Assign[$"b_idx"$][$j "mod" k$]
        Assign[$Z$][$A_i dot "front" dot "w" + B_("b_idx") dot "back" dot "w"$]
        Assign[$Y$][$A_i dot "back" dot "w" + B_("b_idx") dot "front" dot "w"$]
        If($Y <= c$, {
          Comment[Tutte le coppie nel blocco sono valide]
          If($A_i dot "maxValue" + B_("b_idx") dot "maxValue" > "best_value"$, {
            Assign[$"best_value"$][$A_i dot "maxValue" + B_("b_idx") dot "maxValue"$]
          })
          states[$"local_results" dot "add"(A_i, B_("b_idx"))$]
        })
        ElseIf(
          $Z <= c space "and" space Y > c$,
          {
            Comment[Alcune coppie potrebbero essere valide]
            states[$"local_results" dot "add"(A_i, B_("b_idx"))$]
          },
          {
            Comment[$Z > c$: nessuna coppia valida, pota il blocco]
            Line[*skip*]
          },
        )
      })
    })
    Line[$"remaining" <- union_(i = 1)^k "local_results"_i$]
  },

) <alg:parallel_pruning>

L'algoritmo di pruning sfrutta il fatto che i blocchi sono ordinati per peso. Per ogni coppia di blocchi $(A_i, B_j)$, vengono calcolati due valori chiave:
- $Z = A_i."front".w + B_j."back".w$: il peso minimo possibile combinando il primo elemento di $A_i$ con l'ultimo di $B_j$
- $Y = A_i."back".w + B_j."front".w$: il peso massimo possibile combinando l'ultimo elemento di $A_i$ con il primo di $B_j$

La logica di potatura è la seguente:

+ Se $Y <= c$, tutte le combinazioni tra i due blocchi rispettano il vincolo di capacità. La coppia di blocchi viene mantenuta e il loro massimo profitto combinato viene aggiornato.
+ Se $Z <= c < Y$, alcune combinazioni potrebbero essere valide mentre altre no. La coppia di blocchi viene mantenuta per un'analisi più approfondita.
+ Se $Z > c$, nessuna combinazione tra i due blocchi può rispettare il vincolo di capacità. La coppia viene scartata (potata).

A questo punto è possibile passare alla ricerca parallela della soluzione. L'algoritmo si compone di due sotto-fasi: il secondo salvataggio parallelo dei massimi (Algoritmo 5 del paper COPA) e la ricerca parallela vera e propria (Algoritmo 6).

=== Secondo salvataggio parallelo dei massimi

Dopo il pruning, al più $2k - 1$ coppie di blocchi vengono distribuite equamente tra i $k$ processori: ogni processore $P_i$ gestisce al massimo $s <= 2$ coppie. Per ogni blocco $B_("it")$ assegnato, si calcola un array di #emph[*suffix max*]: $"Max"_(i:j)[t]$ rappresenta il profitto massimo tra tutti gli elementi da posizione $j$ a $e$ del blocco $B_("it")$, mentre $L^t_j$ rappresenta l'indice del massimo trovato.

#algorithm-figure(
  "Secondo salvataggio parallelo dei massimi (suffix max)",
  {
    import algorithmic: *
    reqline[Coppie di blocchi rimanenti dopo il pruning]
    ensureline[$"Max"_(i:j)[t]$ e $L^t_j$ per ogni blocco $B_("it")$]
    For([#strong[all] $P_i$, $1 <= i <= k$, #strong[in parallel]], {
      For($t <- 0 space "to" space s - 1$, {
        states[$"Max"_(i:e)[t] <- b_("it":e) dot "p"$, $L^t_e <- e$]
        For($j <- e - 1 space "downto" space 1$, {
          If(
            $b_("it":j) dot "p" > "Max"_(i:(j+1))[t]$,
            {
              states[$"Max"_(i:j)[t] <- b_("it":j) dot "p"$, $L^t_j <- j$]
            },
            {
              states[$"Max"_(i:j)[t] <- "Max"_(i:(j+1))[t]$, $L^t_j <- L^t_(j+1)$]
            },
          )
        })
      })
    })
  },

) <alg:parallel_suffix_max>

Questo pre-calcolo consente, durante la ricerca, di conoscere in $O(1)$ il miglior profitto ottenibile da un dato punto in poi nella lista $B$, accelerando notevolmente la scansione.

=== Ricerca parallela con two-pointer

Ciascun processore esegue una ricerca #emph[*two-pointer*] sulle proprie coppie di blocchi $(A_i, B_("it"))$. Per ogni coppia, due puntatori $x$ (su $A_i$) e $y$ (su $B_("it")$) scorrono i blocchi in direzioni opposte:

#algorithm-figure(
  "Ricerca parallela (two-pointer search)",
  {
    import algorithmic: *
    reqline[Coppie di blocchi rimanenti dopo il pruning, suffix max arrays]
    ensureline[Soluzione finale $"Bestvalue"$ e $X$]
    For([#strong[all] $P_i$, $1 <= i <= k$, #strong[in parallel]], {
      states[$t <- 0$, $"Maxvalue"_i <- 0$, $X_i <- (0, 0)$]
      While($0 <= t <= s - 1$, {
        Line[Seleziona la coppia di blocchi $(A_i, B_("it"))$]
        states[$x <- 1$, $y <- 1$]
        While($x <= e space "and" space y <= e$, {
          If($a_(i:x) dot "w" + b_("it":y) dot "w" > c$, {
            states[$y <- y + 1 space "and" space "continue"$]
          })
          If($a_(i:x) dot "p" + "Max"_(i:y)[t] > "Maxvalue"_i$, {
            Assign[$"Maxvalue"_i$][$a_(i:x) dot "p" + "Max"_(i:y)[t]$]
            Assign[$X_i$][$(a_(i:x), b_("it":L^t_y))$]
          })
          Assign[$x$][$x + 1$]
        })
        Assign[$t$][$t + 1$]
      })
    })
    Line[*Riduzione globale:* $"Bestvalue" <- "Maxvalue"_1$, $X <- X_1$]
    For($i <- 2 space "to" space k$, {
      If($"Bestvalue" < "Maxvalue"_i$, {
        states[$"Bestvalue" <- "Maxvalue"_i$, $X <- X_i$]
      })
    })
    Line[Converti le componenti di $X$ in binario e concatena per ottenere la soluzione finale]
  },

) <alg:parallel_search>

La logica è la seguente: poiché $A_i$ è ordinato per peso crescente e $B_("it")$ per peso decrescente, quando la somma dei pesi supera la capacità $c$, si incrementa $y$ per passare a un elemento più leggero in $B$. Quando invece la somma è ammissibile, il profitto candidato si calcola come $a_(i:x) dot "p" + "Max"_(i:y)[t]$, dove $"Max"_(i:y)[t]$ rappresenta il miglior profitto ottenibile da $y$ in poi nel blocco $B_("it")$. Questo evita di dover iterare su tutte le coppie di elementi, riducendo la complessità a $O(e) = O(N/k)$.

La complessità complessiva dello stage di ricerca parallela è $O(4N\/k + k)$, che nel caso $k = O(N^(1\/2))$ diventa $O(2^(n\/4))$.

= COPA: Cost Optimal Parallel Algorithm

Un interessante modello di esecuzione è denominato COPA @knapsackCopaSolution e sviluppa un algoritmo parallelo cost optimal per
risolvere il problema sia su CPU con OpenMP che su GPU con CUDA, l'idea dietro quest'algoritmo è la seguente: 1) Si prende
l'array di oggetti $V$ e lo si divide in 2 insiemi disgiunti. 2) Per il primo insieme si generano tutte le $N = 2^(n\/2)$ possibili
combinazioni $a_i$, si calcola di ogni sottoinsieme $a_i$ peso e profitto $w_i, p_i$ e si aggiunge questa tripletta in una lista
in ordine crescente. 3) Si esegue lo stesso con la seconda lista per ogni sottoinsieme $b_i$ e le triplette si mettono in una lista
di ordine non crescente. In questo modo si è generata la lista A, per generare la lista B si riproducono gli step precedenti
semplicemente inserendo le triplette in una lista di ordine non crescente. A questo punto si può passare alla fase di ricerca
che prenderà in input le due liste ordinate.

#algorithm-figure(
  "Algoritmo di ricerca",
  {
    import algorithmic: *
    reqline[Due liste ordinate $A$ e $B$]
    ensureline[Soluzione finale $"Bestvalue"$ e $X$]
    Line[Inizializza $"MaxB"_N <- b_N dot "p"$, $"L"_N <- N$, $"Bestvalue" <- 0$, $"X"_1 <- (0, 0)$]
    For($"i" <- N - 1 space "downto" space 1$, {
      If(
        $b_i dot "p" > "MaxB"_(i+1)$,
        {
          states[$"MaxB"_i <- b_i dot "p" space "and" space "L"_i <- i$]
        },
        {
          states[$"MaxB"_i <- "MaxB"_(i+1) space "and" space "L"_i <- "L"_(i+1)$]
        },
      )
    })
    states[$i <- 1 space "and" space j <- 1$]
    While($i <= N space "and" space j <= N$, {
      If($a_i dot "w" + b_j dot "w" > c$, {
        states[$j <- j + 1 space "and" space "continue"$]
      })
      If($a_i dot "p" + "MaxB"_j > "Bestvalue"$, {
        states[$"Bestvalue" <- a_i dot "p" + "MaxB"_j space "and" space "X"_1 <- (a_i, b_(L_j))$]
      })
      Assign[$i$][$i + 1$]
    })
    Line[Converti le due componenti decimali di $X_1$ in due numeri binari: $X_1 <- ("binary"_1, "binary"_2)$]
    LineComment(Assign[$X$][Call["strcat"][$"binary"_1$, $"binary"_2$]], [Concatena i due binari])
  },

) <alg:cap>


Questo modello di esecuzione si può vedere subito che si presta bene ad essere parallelizzato, sia nella parte di generazione
delle liste sia in quella poi di ricerca dell'ottimo.



L'algoritmo parallelo è stato pensato per essere risolto su un modello EREW (Exclusive Read, Exclusive Write) PRAM con shared memory e assume $k$ processori accessibili.
L'algoritmo sfrutta prima un'algoritmo di generazione parallela delle liste A e B

#algorithm-figure(
  "Algoritmo parallelo di generazione",
  {
    import algorithmic: *
    reqline[Elementi $(v_i, v_i dot "w", v_i dot "p")$, $i = 1, 2, dots, n/2$]
    ensureline[Lista ordinata $A_(n/2)$ (i.e. $A$)]
    Line[Inizializza $A_0 <- [(0, 0, 0)]$ e $"IC" <- 1$]
    For($i <- 0 space "to" space n/2 - 1$, {
      Assign[$v_(i+1)$][$"IC"$]
      For([#strong[all] $P_j$, $1 <= j <= k$, #strong[in parallel]], {
        Line[Genera nuova lista $A'_i$ aggiungendo la tripletta $(v_(i+1), v_(i+1) dot "w", v_(i+1) dot "p")$ a tutti gli elementi della lista ordinata $A_i$]
        Line[Chiama l'algoritmo di merge parallelo per unire $A_i$ e $A'_i$ in ordine non decrescente di peso, producendo la lista ordinata $A_(i+1)$]
      })
      Assign[$"IC"$][$"IC" + 1$]
    })
  },

) <alg:parallel_gen>

l'algoritmo di merging parallelo si articola nel modo seguente: prende in input due vettori ordinati $U = (u_1, u_2, dots, u_m)$ e $V = (v_1, v_2, dots, v_m)$ e $k$ processori $P_1, P_2, dots, P_k$, con $1 <= k <= 2m$ potenza di 2, e produce un vettore ordinato di lunghezza $2m$ ottenuto dall'unione di $U$ e $V$.

+ #text(weight: "bold")[Step 1:] I $k$ processori partizionano $U$ e $V$ in parallelo, ciascuno in $k$ sottovettori (che possono essere vuoti) $(U_1, U_2, dots, U_k)$ e $(V_1, V_2, dots, V_k)$ tali che: $|U_i| + |V_i| = 2m\/k$ per $1 <= i <= k$, e i pesi di tutti gli elementi in $U_i$ e $V_i$ sono minori dei pesi di tutti gli elementi in $U_(i+1)$ e $V_(i+1)$, per $1 <= i <= k - 1$.
+ #text(weight: "bold")[Step 2:] Il processore $P_i$ esegue il merge sequenziale di $U_i$ e $V_i$, per $1 <= i <= k$, in parallelo con gli altri processori.

A questo punto si può passare a dividere equamente tra i processori disponibili i blocchi ottenuti e calcolare per ognuno di questi blocchi gli elementi di profitto massimo.

#algorithm-figure(
  "ALgoritmo di divisione e ricerca del massimo profitto per blocco",
  {
    import algorithmic: *
    reqline[Due liste ordinate $A$ e $B$]
    ensureline[$"MaxA"$ e $"MaxB"$]
    Line[Dividi sia $A$ che $B$ uniformemente in $k$ blocchi]
    For([#strong[all] $P_i$, $1 <= i <= k$, #strong[in parallel]], {
      states[$"MaxA"_i <- a_(i,1) dot "p"$, $"MaxB"_i <- b_(i,1) dot "p"$]
      For($j <- 2 space "to" space e$, {
        If($a_(i,j) dot "p" > "MaxA"_i$, {
          Assign[$"MaxA"_i$][$a_(i,j) dot "p"$]
        })
        If($b_(i,j) dot "p" > "MaxB"_i$, {
          Assign[$"MaxB"_i$][$b_(i,j) dot "p"$]
        })
      })
    })
  },

) <alg:parallel_max>

Ora lo spazio delle soluzioni è davvero molto grande, per questo motivo si ricorre a un algoritmo di potatura dell'insieme delle soluzioni

#algorithm-figure(
  "Algoritmo di pruning parallelo",
  {
    import algorithmic: *
    reqline[Due liste di blocchi $A = {A_1, A_2, dots, A_k}$ e $B = {B_1, B_2, dots, B_k}$, capacità $c$]
    ensureline[Insieme di coppie di blocchi non potate $"remaining"$]
    Line[Inizializza $"remaining" <- emptyset$]
    For([#strong[all] $P_i$, $1 <= i <= k$, #strong[in parallel]], {
      Line[Inizializza $"local_results" <- emptyset$]
      For($j <- i space "to" space k + i - 1$, {
        Assign[$"b_idx"$][$j "mod" k$]
        Assign[$Z$][$A_i dot "front" dot "w" + B_("b_idx") dot "back" dot "w"$]
        Assign[$Y$][$A_i dot "back" dot "w" + B_("b_idx") dot "front" dot "w"$]
        If($Y <= c$, {
          Comment[Tutte le coppie nel blocco sono valide]
          If($A_i dot "maxValue" + B_("b_idx") dot "maxValue" > "best_value"$, {
            Assign[$"best_value"$][$A_i dot "maxValue" + B_("b_idx") dot "maxValue"$]
          })
          states[$"local_results" dot "add"(A_i, B_("b_idx"))$]
        })
        ElseIf(
          $Z <= c space "and" space Y > c$,
          {
            Comment[Alcune coppie potrebbero essere valide]
            states[$"local_results" dot "add"(A_i, B_("b_idx"))$]
          },
          {
            Comment[$Z > c$: nessuna coppia valida, pota il blocco]
            Line[*skip*]
          },
        )
      })
    })
    Line[$"remaining" <- union_(i = 1)^k "local_results"_i$]
  },

) <alg:parallel_pruning>

L'algoritmo di pruning sfrutta il fatto che i blocchi sono ordinati per peso. Per ogni coppia di blocchi $(A_i, B_j)$, vengono calcolati due valori chiave:
- $Z = A_i."front".w + B_j."back".w$: il peso minimo possibile combinando il primo elemento di $A_i$ con l'ultimo di $B_j$
- $Y = A_i."back".w + B_j."front".w$: il peso massimo possibile combinando l'ultimo elemento di $A_i$ con il primo di $B_j$

La logica di potatura è la seguente:

+ Se $Y <= c$, tutte le combinazioni tra i due blocchi rispettano il vincolo di capacità. La coppia di blocchi viene mantenuta e il loro massimo profitto combinato viene aggiornato.
+ Se $Z <= c < Y$, alcune combinazioni potrebbero essere valide mentre altre no. La coppia di blocchi viene mantenuta per un'analisi più approfondita.
+ Se $Z > c$, nessuna combinazione tra i due blocchi può rispettare il vincolo di capacità. La coppia viene scartata (potata).

A questo punto è possibile passare alla ricerca parallela della soluzione. L'algoritmo si compone di due sotto-fasi: il secondo salvataggio parallelo dei massimi (Algoritmo 5 del paper COPA) e la ricerca parallela vera e propria (Algoritmo 6).

=== Secondo salvataggio parallelo dei massimi

Dopo il pruning, al più $2k - 1$ coppie di blocchi vengono distribuite equamente tra i $k$ processori: ogni processore $P_i$ gestisce al massimo $s <= 2$ coppie. Per ogni blocco $B_("it")$ assegnato, si calcola un array di #emph[*suffix max*]: $"Max"_(i:j)[t]$ rappresenta il profitto massimo tra tutti gli elementi da posizione $j$ a $e$ del blocco $B_("it")$, mentre $L^t_j$ rappresenta l'indice del massimo trovato.

#algorithm-figure(
  "Secondo salvataggio parallelo dei massimi (suffix max)",
  {
    import algorithmic: *
    reqline[Coppie di blocchi rimanenti dopo il pruning]
    ensureline[$"Max"_(i:j)[t]$ e $L^t_j$ per ogni blocco $B_("it")$]
    For([#strong[all] $P_i$, $1 <= i <= k$, #strong[in parallel]], {
      For($t <- 0 space "to" space s - 1$, {
        states[$"Max"_(i:e)[t] <- b_("it":e) dot "p"$, $L^t_e <- e$]
        For($j <- e - 1 space "downto" space 1$, {
          If(
            $b_("it":j) dot "p" > "Max"_(i:(j+1))[t]$,
            {
              states[$"Max"_(i:j)[t] <- b_("it":j) dot "p"$, $L^t_j <- j$]
            },
            {
              states[$"Max"_(i:j)[t] <- "Max"_(i:(j+1))[t]$, $L^t_j <- L^t_(j+1)$]
            },
          )
        })
      })
    })
  },

) <alg:parallel_suffix_max>

Questo pre-calcolo consente, durante la ricerca, di conoscere in $O(1)$ il miglior profitto ottenibile da un dato punto in poi nella lista $B$, accelerando notevolmente la scansione.

=== Ricerca parallela con two-pointer

Ciascun processore esegue una ricerca #emph[*two-pointer*] sulle proprie coppie di blocchi $(A_i, B_("it"))$. Per ogni coppia, due puntatori $x$ (su $A_i$) e $y$ (su $B_("it")$) scorrono i blocchi in direzioni opposte:

#algorithm-figure(
  "Ricerca parallela (two-pointer search)",
  {
    import algorithmic: *
    reqline[Coppie di blocchi rimanenti dopo il pruning, suffix max arrays]
    ensureline[Soluzione finale $"Bestvalue"$ e $X$]
    For([#strong[all] $P_i$, $1 <= i <= k$, #strong[in parallel]], {
      states[$t <- 0$, $"Maxvalue"_i <- 0$, $X_i <- (0, 0)$]
      While($0 <= t <= s - 1$, {
        Line[Seleziona la coppia di blocchi $(A_i, B_("it"))$]
        states[$x <- 1$, $y <- 1$]
        While($x <= e space "and" space y <= e$, {
          If($a_(i:x) dot "w" + b_("it":y) dot "w" > c$, {
            states[$y <- y + 1 space "and" space "continue"$]
          })
          If($a_(i:x) dot "p" + "Max"_(i:y)[t] > "Maxvalue"_i$, {
            Assign[$"Maxvalue"_i$][$a_(i:x) dot "p" + "Max"_(i:y)[t]$]
            Assign[$X_i$][$(a_(i:x), b_("it":L^t_y))$]
          })
          Assign[$x$][$x + 1$]
        })
        Assign[$t$][$t + 1$]
      })
    })
    Line[*Riduzione globale:* $"Bestvalue" <- "Maxvalue"_1$, $X <- X_1$]
    For($i <- 2 space "to" space k$, {
      If($"Bestvalue" < "Maxvalue"_i$, {
        states[$"Bestvalue" <- "Maxvalue"_i$, $X <- X_i$]
      })
    })
    Line[Converti le componenti di $X$ in binario e concatena per ottenere la soluzione finale]
  },

) <alg:parallel_search>

La logica è la seguente: poiché $A_i$ è ordinato per peso crescente e $B_("it")$ per peso decrescente, quando la somma dei pesi supera la capacità $c$, si incrementa $y$ per passare a un elemento più leggero in $B$. Quando invece la somma è ammissibile, il profitto candidato si calcola come $a_(i:x) dot "p" + "Max"_(i:y)[t]$, dove $"Max"_(i:y)[t]$ rappresenta il miglior profitto ottenibile da $y$ in poi nel blocco $B_("it")$. Questo evita di dover iterare su tutte le coppie di elementi, riducendo la complessità a $O(e) = O(N/k)$.

La complessità complessiva dello stage di ricerca parallela è $O(4N\/k + k)$, che nel caso $k = O(N^(1\/2))$ diventa $O(2^(n\/4))$.


