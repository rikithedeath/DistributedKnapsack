#import "../functions/preamble.typ": *
#import "@preview/algorithmic:1.0.7"
#import algorithmic: Line, algorithm-figure, style-algorithm



L'algoritmo dello zaino è un noto problema di ottimizzazione. L'idea è che si ha a disposizione uno zaino
e diversi elementi ad ognuno dei quali è assegnato un peso e un profitto. L'obiettivo è di massimizzare il profitto mantenendo il peso
totale degli elementi entro un limite prefissato. Nella versione detta 0/1, ogni elemento può essere preso una sola volta,
il problema si pone quindi come
#align(center)[$ op("max") sum v_i x_i $]
dove $v_i$ è il profitto dell'oggetto i-esimo e $x_i$ rappresenta la scelta (1 se scelto 0 altrimenti), soggetto a
#align(center)[$ sum w_i x_i <= W space and space x_i in {0, 1} $]

Le versioni più complete di questo problema prevedono che si possa scegliere lo stesso elemento più volte, sia con che senza limite superiore a queste scelte
(rispettivamente Bounded o Unbounded). Esistono anche versioni più complesse, come lo zaino multidimensionale, dove non esiste un solo vincolo
(come nel caso più semplice il peso) ma esistono diversi vincoli da tenere in considerazione durante il calcolo. Questa classe di problemi gioca
un ruolo chiave in diversi campi molto delicati ad esempio nelle operazioni logistiche o nello scheduling di task satellitari, in cui infatti
stanno prendendo piede diverse soluzioni basate su euristiche.

== Soluzioni per knapsack 0/1

Il problema è NP-completo, data una soluzione non esiste quindi un algoritmo che possa verificare se è ottima in tempo polinomiale.
Esiste tuttavia una soluzione che lo risolve in tempo pseudo-polinomiale basata sulla programmazione dinamica.
La prima soluzione più semplice e meno ottimale ha complessità temporale $O(2^n)$ ed è basata su un albero decisionale esplorato ricorsivamente

#algorithm-figure(
  "Knapsack 0/1 Ricorsivo",
  {
    import algorithmic: *
    reqline[$"profits"$, $"weights"$, $"max_weight"$, $"index"$]
    ensureline[Profitto massimo ottenibile]
    If($"index" = 0 space "or" space "max_weight" = 0$, {
      Return[$0$]
    })
    Assign[$"pick"$][$0$]
    If($"weights"["index" - 1] <= "max_weight"$, {
      Assign[$"pick"$][$"profits"["index" - 1] + Call["Knapsack"][$"max_weight" - "weights"["index" - 1]$,$"profits"$,$"weights"$,$"index" - 1$]$]
    })
    Assign[$"not_pick"$][$#Call[Knapsack][$"max_weight"$, $"profits"$, $"weights"$, $"index" - 1$]$]
    Return[op("max")($"pick"$, $"not_pick"$)]
  },
) <alg:knapsack-rec>

Le altre due soluzioni sono basate sulla programmazione dinamica.
la prima usa un approccio top-down (memoization) proponendo sostanzialmente una naturale evoluzione dell'approccio precedente,
ha una complessità temporale e spaziale $O(n * W)$

#algorithm-figure(
  "Knapsack 0/1 Top-Down (memoization)",
  {
    import algorithmic: *
    reqline[$"profits"$, $"weights"$, $"max_weight"$, $"index"$, $"table"$]
    ensureline[Profitto massimo ottenibile]
    If($"index" = 0 space "or" space "max_weight" = 0$, {
      Return[$0$]
    })
    If($"table"["index"]["max_weight"] != -1$, {
      Return[$"table"["index"]["max_weight"]$]
    })
    Assign[$"pick"$][$0$]
    If($"weights"["index" - 1] <= "max_weight"$, {
      Assign[$"pick"$][$"profits"["index" - 1] + #Call[Knapsack][$"max_weight" - "weights"["index" - 1]$, $"profits"$, $"weights"$, $"index" - 1$, $"table"$]$]
    })
    Assign[$"not_pick"$][$#Call[Knapsack][$"max_weight"$, $"profits"$, $"weights"$, $"index" - 1$, $"table"$]$]
    Assign[$"table"["index"]["max_weight"]$][op("max")($"pick"$, $"not_pick"$)]
    Return[$"table"["index"]["max_weight"]$]
  },

) <alg:knapsack-memo>

la seconda è una versione bottom-up (tabulation) con la medesima complessità ma che evita la ricorsione

#algorithm-figure(
  "Knapsack 0/1 Bottom-Up (tabulation)",
  {
    import algorithmic: *
    reqline[$"profits"$, $"weights"$, $"max_weight"$, $"len"$]
    ensureline[Profitto massimo ottenibile]
    Assign[$"table"$][Call[Matrix][$"len" + 1$, $"max_weight" + 1$]]
    For($"i" <- 0 space "to" space "len"$, {
      For($"j" <- 0 space "to" space "max_weight"$, {
        If(
          $"i" = 0 space "or" space "j" = 0$,
          {
            Assign[$"table"[i][j]$][$0$]
          },
          {
            Assign[$"pick"$][$0$]
            If($"weights"[i - 1] <= j$, {
              Assign[$"pick"$][$"profits"[i - 1] + "table"[i - 1][j - "weights"[i - 1]]$]
            })
            Assign[$"not_pick"$][$"table"[i - 1][j]$]
            Assign[$"table"[i][j]$][op("max")($"pick"$, $"not_pick"$)]
          },
        )
      })
    })
    Return[$"table"["len"]["max_weight"]$]
  },

) <alg:knapsack-tab>

Una soluzione più raffinata di questa prevede di usare solo l'ultima riga calcolata, in modo da ridurre la complessità spaziale a $O(W)$

#algorithm-figure(
  "Knapsack 0/1 Bottom-Up (spazio ottimizzato)",
  {
    import algorithmic: *
    reqline[$"profits"$, $"weights"$, $"max_weight"$, $"len"$]
    ensureline[Profitto massimo ottenibile]
    Assign[$"table"$][Call[array][$0 dots "max_weight"$] inizializzato a 0]
    For($"i" <- 1 space "to" space "len"$, {
      For($"j" <- "max_weight" space "downto" space "weights"[i - 1]$, {
        Assign[$"table"[j]$][op("max")($"table"[j]$, $"table"[j - "weights"[i - 1]] + "profits"[i - 1]$)]
      })
    })
    Return[$"table"["max_weight"]$]
  },

) <alg:knapsack-opt>

Lo svantaggio principale delle soluzioni basate su programmazione dinamica è che non lavorano bene su casi in cui i pesi non sono interi
(o non possiedono una differenza prefissata tra di loro). Lo stesso vale per l'approccio naive descritto nella prima soluzione.

== Soluzione Greedy

La soluzione Greedy ha un indubbio vantaggio rispetto a tutte le altre, può scegliere anche frazioni di un oggetto,
l'idea è quella di prendere gli oggetti basandosi sul loro rapporto peso/valore, prendendoli in ordine decrescente di rapporto,
finchè lo zaino non è pieno. Se un oggetto non è possibile prenderlo per intero allora si può prendere una sua frazione a seconda
della capacità rimanente.

#algorithm-figure(
  "Fractional Knapsack (Greedy)",
  {
    import algorithmic: *
    reqline[$"profits"$, $"weights"$, $"max_weight"$, $"len"$]
    ensureline[Profitto massimo ottenibile]
    Assign[$"items"$][Call[sort][Call[zip][$"profits"$, $"weights"$], key = $("profit")/("weight")$ *decrescente*]]
    Assign[$"profit"$][$0.0$]
    Assign[$"current_capacity"$][$"max_weight"$]
    For($"i" <- 0 space "to" space "len" - 1$, {
      If(
        $"items"[i] dot "weight" <= "current_capacity"$,
        {
          Assign[$"profit"$][$"profit" + "items"[i] dot "profit"$]
          Assign[$"current_capacity"$][$"current_capacity" - "items"[i] dot "weight"$]
        },
        {
          Assign[$"profit"$][$"profit" + ("items"[i] dot "profit")/("items"[i] dot "weight") times "current_capacity"$]
          Break
        },
      )
    })
    Return[$"profit"$]
  },

) <alg:fknapsack>


