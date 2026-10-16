
#import "@preview/algorithmic:1.0.7"
#import algorithmic: Line, algorithm-figure, iflike, style-algorithm
#import "@preview/codelst:2.0.2": sourcecode



#let ParallelFor = iflike.with(kw1: "for", kw2: "do in parallel", kw3: "end")

#let dp_algo_dag_sequential = algorithm-figure(
  "Tiled Knapsack DP",
  inset: 0.42em,
  {
    import algorithmic: *
    let AddVertex = Call.with("AddVertex")
    Procedure(
      "Costruzione DAG",
      ($n$, $C$, $w$, $v$, $"item_block"$, $"cap_block"$),
      {
        Comment[Inizializzazione]
        Assign[$"nb"$][$floor((n+"item_block"- 1)/ "item_block")$]
        Assign[$"nq"$][$floor((C+ "cap_block")/"cap_block")$]

        Comment[Costruzione del DAG, $"vertex"(b, q) = b dot "nq" + q$]
        For($b <- 1 space "to" space "nb"$, {
          For($q <- 0 space "to" space "nq" - 1$, {
            AddVertex[$(b - 1, q) -> (b, q)$]
            For($"item" space i space in space "blocco b"$, {
              Assign[$"lo"$][$min(q dot "cap_block" - w_i, 0)$]
              Assign[$"hi"$][$max((q + 1) dot "cap_block" - 1 - w_i, C)$]
              For($"column qp" in ("lo", "hi")$, {
                If(
                  $"qp" < "q"$,
                  {
                    AddVertex[$(b, "qp") -> (b, q)$]
                    AddVertex[$(b - 1, "qp") -> (b, q)$]
                  },
                )
              })
            })
          })
        })

        Procedure(
          "Risoluzione DAG",
          ($G$, $C$),
          {
            For($"tile" (b, q) space "in ordine row-major"$, {
              Assign[$"block"$][matrice della tile con $"rows" = "item_block" + 1$]
              If(
                $"b" == 0$,
                {
                  Assign[$"block"(0, c)$][0]
                },
                {
                  For($"ogni colonna" c$, {
                    Assign[$"wp"$][$q dot "cap_block" + c$]
                    Assign[$"block"(0, c)$][$"tile"(b - 1, "wp"/"cap_block")."block"("rows_local", "wp" mod "cap_block")$]
                  })
                },
              )
              For($"item" space a space "dove" space i = b dot "item_block" + a$, {
                For($"ogni colonna" c$, {
                  Assign[$"wp"$][$q dot "cap_block" + c$]
                  Assign[$"v"$][$"block"(a, c)$]
                  If(
                    $"w"_i <= "wp"$,
                    {
                      Assign[$"src"$][$"wp" - "w"_i$]
                      IfElseChain(
                        $"src" >= "lo"$,
                        {
                          Assign[$"candidate"$][$"block"(a, "src" - "lo") + v_i$]
                        },
                        {
                          Assign[$"candidate"$][$"tile"(b, "src"/"cap_block")."block"(a, "src" mod "cap_block") + v_i$]
                        },
                      )
                      Assign[$"v"$][$"max"("v", "candidate")$]
                    },
                  )
                  Assign[$"block"(a + 1, c)$][$v$]
                })
              })
            })
            Assign[$"opt"$][$"tile"("nb" - 1, "nq" - 1)."block"("rows" - 1, "C" - ("nq" - 1) dot "cap_block")$]
          },
        )
        Procedure("Backtrack items", ($"dp"$, $C$), {
          Comment[Ricostruzione degli item]
          Assign[$"items"$][$emptyset$]
          Assign[$"wp"$][$C$]
          For($"i" <- n space "downto" space 1$, {
            If(
              $"dp"(i, "wp") != "dp"(i - 1, "wp")$,
              {
                Assign[$"items"$][$"items" union {i - 1}$]
                Assign[$"wp"$][$"wp" - w[i - 1]$]
              },
            )
          })
          Return[$"items , opt"$]
        })
      },
    )
  },
  supplement: none,
);

#let KpDp = algorithm-figure("Knapsack DP: Naive frontier parallel", {
  import algorithmic: *
  Procedure("Knapsack DP", ("w:[]", "v:[]", "C: int"), {
    Assign[$n$][$"len(w)"$]
    Assign[$"dp"$][$"[][]"$]
    ParallelFor($i in "(0..n)"$, {
      IfElseChain(
        $"weights"[i-1] <= w$,
        {
          Assign[$"dp[i][w]"$][$"max(dp[i-1][w],dp[i-1][w-weights[i-1]] + values[i-1])"$]
        },
        {
          Assign[$"dp[i][w]"$][$"dp[i-1][w]"$]
        },
      )
    })
  })
})

#let KpDpDag = algorithm-figure(
  "KnapsackDPDAG",
  {
    import algorithmic: *
    let ww = "weights"
    let vv = "values"
    let cc = "capacity"
    let ib = "item_block"
    let cb = "cap_block"
    Procedure("KnapsackDPDAG", ("weights:[]", "values:[]", "capacity", "item_block", "cap_block"), {
      Assign[$g$][#CallInline("GenerateDAG", [$ww,vv,cc,ib,cb$])]
      Assign[$n$][#CallInline[$"len"$][$w$]]
      Comment[I vertici del DAG sono in ordine topologico]
      Assign[$"levels"$][#CallInline("AssignLevels", [$"DAG"$])]
      For($v in g$, {
        Assign[$L$][$"level[v]"$]
        Call[Append][$"by_level[L]"$,$v$]
      })
      For($l in "(0..L)"$, {
        ParallelFor($v in "by_level[l]"$, {
          CallInline("ComputeTile", "v")
        })
      })
      Return[$"DP[n][capacity]"$]
    })
  },
)


#let GenerateDAG = algorithm-figure(
  "KnapsackDPDAG.GenerateDAG",
  inset: (x: 0.2em, y: 0.42em),
  {
    import algorithmic: *
    let AddVertex = Call.with("AddVertex")
    let AddEdge = Call.with("AddEdge")
    Procedure("GenerateDAG", ("weights:[]", "capacity", "item_block", "cap_block"), {
      Assign[$"nb"$][$ceil(n/"item_block")$]
      Assign[$"nq"$][$ceil(("capacity"+1)/"cap_block")$]
      Assign[$g$][$"Graph{}"$]
      Comment[Fase 1: creazione vertici]
      For($b in "(0..nb-1)"$, {
        For($q in "(0..nq-1)"$, {
          AddVertex[$(b,q)$]
        })
      })
      For($b in "(0..nb-1)"$, {
        For($q in "(0..nq-1)"$, {
          Comment[La line è l'indice della linea all'interno della DP]
          For($"line" in (b,q)$, {
            If($exists w in "line" | exists x in "Pred"(w) in.not (b,q)$, {
              Comment[Si aggiunge allora un arco tra il blocco corrente e quello con il predecessore]
              AddEdge[$(m,n) | x in (m,n),(b,q)$]
            })
          })
        })
      })
      Return[$g$]
    })
  },
)

#let AssignLevels = algorithm-figure(
  "KnapsackDPDAG.AssignLevels",
  inset: (x: 0.2em, y: 0.42em),
  {
    import algorithmic: *
    Procedure("KnapsackDPDAG.AssignLevels", "g", {
      Comment[i vertici vengono inseriti in ordine row-major e quindi in ordine topologico]
      For($v in g$, {
        IfElseChain(
          $"Pred(v)" = emptyset$,
          {
            Assign["level[v]"][0]
          },
          {
            Assign["level[v]"][$max("p.level" in "Pred"(v)) + 1$]
          },
        )
      })
      Return[$"level"$]
    })
  },
)

#let ComputeTile = algorithm-figure(
  "KnapsackDPDAG.ComputeTile",
  inset: (x: 0.2em, y: 0.42em),
  {
    import algorithmic: *
    Procedure("KnapsackDPDAG.ComputeTile", ("dp", "tile", "weights", "values"), {
      Assign[$"i_start,i_end"$][$"first-item","last-item"$]
      Assign[$"w_start","w_end"$][$"first-cap","last_cap"$]
      For($i in "(i_start,i_end)"$, {
        Assign[w_i]["weights[i]"]
        Assign[v_i]["values[i]"]
        For($w in "(w_start,w_end)"$, {
          Assign[$"skip"$][$"DP[i-1][w]"$]
          IfElseChain(
            $w >= w_i$,
            {
              Assign[$"take"$][$"DP[i-1][w-w_i]" + v_i$]
              Assign[$"DP[i][w]"$][$max("skip", "take")$]
            },
            {
              Assign[$"DP[i][w]"$][$"skip"$]
            },
          )
        })
      })
    })
  },
)
