#import "../functions/preamble.typ": *


l'esperimento si divide in 4 parti distinte: 1) l'implementazione del algoritmo di risoluzione del knapsack con programmazione dinamica: questo metodo è quello più semplice da implementare ed è stato usato come base per verificare la corretta implementazione negli unit test degli algoritmi successivi. 2) Implementazione dell'algoritmo con programmazione dinamica in MPI 3) Implementazione dell'algoritmo COPA sequenziale e parallelo in shared memory usando OpenMP. 4) Implementazione dell'algoritmo COPA distribuito usando MPI

#include "knapsack_dp_classic.typ"

== Knapsack COPA

La versione COPA dell'algoritmo è stata pensata per essere eseguita in shared memory e per ovviare a un problema della versione in dynamic programming dell'algoritmo, cioè che la grandezza della tabella cresce con l'aumentare della capacità dello zaino. Invece nella versione COPA il numero di soluzioni da esplorare è perlopiù legato al numero degli elementi, tuttavia questo insieme cresce con un ritmo di $2^(n\/2)$, quindi per un numero di elementi superiore a 40-50 il numero di combinazioni diventa proibitivo da memorizzare (come anche sottolineato nell'articolo originale).

=== Algoritmo di generazione dei sottoinsiemi

I sottoinsiemi sono generati partendo dalla lista generale, dividendola in due e generando a partire dalle due liste spezzate un insieme di sottoinsiemi di soluzioni. L'algoritmo si articola in due fasi: 1) copia della lista precedente e aggiunta dell'elemento a ogni insieme 2) merging della nuova lista con quella precedente usando l'algoritmo di coranking.

#figure(
  sourcecode[
    ```cpp
      using namespace std::ranges;
      int total_size = static_cast<int>(A.size() + B.size());

      #pragma omp parallel for num_threads(num_threads)
      for (int t = 0; t < num_threads; ++t)
      {
          int start = t * total_size / num_threads;
          int end = (t + 1) * total_size / num_threads;

          auto [a_start, b_start] = co_rank(A, B, start);
          auto [a_end, b_end] = co_rank(A, B, end);
          std::merge(begin(A) + a_start, begin(A) + a_end, begin(B) + b_start, begin(B) + b_end, begin(output) + start);
      }
    ```
  ],
  caption: "funzione di Parallel Merging",
  kind: "listing",
)

#figure(
  sourcecode[
    ```cpp
      for (int item_idx = 0; auto &&item : r)
      {
          std::vector<CopaSubset> shifted{subsets.size()};
          auto [w, v] = item;
          for (int i = 0; i < static_cast<int>(shifted.size()); i++)
          {
              shifted[i] = subsets[i];
              shifted[i].addItem(item_idx, w, v);
          }
          item_idx++;
          std::vector<CopaSubset> merged;
          merged.resize(subsets.size() + shifted.size());
          parallel_merge(subsets, shifted, merged, numthreads);
          subsets = std::move(merged);
      }
    ```
  ],
  caption: "Funzione di generazione e merging",
  kind: "listing",
)

Per aumentare il più possibile la località dei dati durante questa fase, si copia il dato e gli si aggiunge l'item direttamente nelle istruzioni successive. Questa funzione nello specifico restituisce il metodo di generazione della lista A in ordine crescente, per la lista B è sufficiente usare std::reverse, una possibile ottimizzazione sarebbe quella di usare std::ranges::reverse in modo che il reversing dei dati sia lazy, cioè effettuato solo quando si pesca dal range. Tuttavia visto che le funzioni successive richiederanno l'uso di input range, che richiede che il range in entrata sia contiguo, questa operazione non sarà possibile poichè il reverse range non è contiguo per lo standard c++. A questo punto si può procedere e suddividere la lista di blocchi in sottoinsiemi di blocchi equamente distribuiti tra i processori, avendo cura di calcolare il massimo profitto durante questa operazione.

#figure(
  sourcecode[
    ```cpp
      int n = static_cast<int>(std::ranges::size(input));
      int k = num_threads;
      int block_size = n / k;
      int remainder = n % k;
      #pragma omp parallel for num_threads(num_threads)
      for (int i = 0; i < k; ++i)
      {
          int start = i * block_size + std::min(i, remainder);
          int end = start + block_size + (i < remainder ? 1 : 0);
          if (start >= end)
          {
              output[i] = {std::span<const CopaSubset>{}, 0};
              continue;
          }
          auto span = std::ranges::subrange(std::ranges::begin(input) + start, std::ranges::begin(input) + end);
          int max_val = 0;
          for (const auto &elem : span)
          {
              if (elem.totalValue > max_val)
                  max_val = elem.totalValue;
          }
          output[i] = {span, max_val};
      }
    ```
  ],
  caption: "Funzione di calcolo del massimo e distribuzione dei blocchi",
  kind: "listing",
)

A questo punto ci si ritrova in output una lista di sottoinsiemi di blocchi, ogni elemento corrisponde al sottoinsieme su cui ci si aspetta che il processo p operi.

=== Pruning delle soluzioni

Una volta compiuta l'operazione di distribuzione si può passare a eseguire il pruning delle soluzioni.

#figure(
  sourcecode[
    ```cpp
      int k = static_cast<int>(blocksB.size());
      int best_value = 0;
      for (int j = i; j < k + i; ++j)
      {
          int b_idx = j % k;
          const auto &blockB = blocksB[b_idx];
          if (blockB.block.empty())
              continue;
          int Z = blockA.block.front().totalWeight + blockB.block.back().totalWeight;
          int Y = blockA.block.back().totalWeight + blockB.block.front().totalWeight;

          // Note: prune is done by not adding the pair to local_results
          if (Y <= capacity)
          {
              // All pairs in this block pair are valid; save max profit and prune
              if (blockA.maxValue + blockB.maxValue > best_value)
              {
                  best_value = blockA.maxValue + blockB.maxValue;
              }
              remaining.emplace_back(blockA, blockB);
          }
          else if (Z <= capacity && Y > capacity)
          {
              remaining.emplace_back(blockA, blockB);
          }
          else if (Z > capacity)
          {
              // No pairs in this block pair are valid; prune
          }
      }
    ```
  ],
  caption: "Funzione di pruning di un sottoinsieme di blocchi",
  kind: "listing",
)

La suddetta funzione deve essere chiamata in modo parallelo in questo modo. La suddivisione è necessaria poichè questa implementazione della funzione di pruning è stata riutilizzata anche nella versione MPI.

#figure(
  sourcecode[
    ```cpp
      std::vector<std::vector<std::pair<CopaBlock, CopaBlock>>> local_results(threads);
      #pragma omp parallel for num_threads(threads)
      for (int i = 0; i < threads; ++i)
      {
          const auto &blockA = blocksA[i];
          prune_block_pair(blockA, blocksB, local_results[i], capacity, i);
      }
      // Merge local results from all processors
      for (const auto &local : local_results)
      {
          remaining.insert(remaining.end(), local.begin(), local.end());
      }
    ```
  ],
  caption: "Chiamata parallela della funzione di pruning",
  kind: "listing",
)

A questo punto si può illustrare la funzione di ricerca parallela dei massimi, che viene chiamata sui blocchi rimanenti in questo modo, anche in questo caso la funzione è stata disaccoppiata poichè la parte di calcolo è stata riutilizzata nella versione a memoria distribuita.

#figure(
  sourcecode[
    ```cpp
      #pragma omp parallel for num_threads(numThreads)
      for (int i = 0; i < numThreads; ++i)
      {
          const auto &[blockA, blockB] = remainingPairs[i];

          // Stage 5: Two-pointer search within this block pair
          BlockPairSearchResult result = block_pair_pointer_search(blockA, blockB, capacity);
          processBestVal[i] = result.bestVal;
          processBestAIdx[i] = result.bestAIdx;
          processBestBIdx[i] = result.bestBIdx;
      }
    ```
  ],
  caption: "Ricerca parallela dei massimi sui blocchi rimanenti",
  kind: "listing",
)

La funzione prende in input le coppie rimanenti in ciascuna lista, che è un vettore di coppie di blocchi, e li esamina in questo modo: prima calcola il suffisso massimo di profitto tra i blocchi della lista B, dopodichè esegue una comparazione con i pesi della lista A.

#figure(
  sourcecode[
    ```cpp
      int blocka_size = static_cast<int>(blockA.block.size());
      int blockb_size = static_cast<int>(blockB.block.size());
      // Stage 4: Suffix max for this B block
      std::vector<int> suffixMaxVal(blockb_size);
      std::vector<int> suffixMaxIdx(blockb_size);
      block_suffix_max_values(blockB.block, suffixMaxVal, suffixMaxIdx);

      int x = 0, y = 0;
      int bestVal = 0, bestA = 0, bestB = 0;
      while (x < blocka_size && y < blockb_size)
      {
          if (blockA.block[x].totalWeight + blockB.block[y].totalWeight > capacity)
          {
              y++;
              continue;
          }
          int candidate = blockA.block[x].totalValue + suffixMaxVal[y];
          if (candidate > bestVal)
          {
              bestVal = candidate;
              bestA = blockA.block[x].index;
              bestB = suffixMaxIdx[y];
          }
          x++;
      }
      return {bestVal, bestA, bestB};
    ```
  ],
  caption: "Funzione di ricerca del massimo tra i blocchi rimanenti",
  kind: "listing",
)

La funzione di calcolo dei suffissi è implementata in questo modo

#figure(
  sourcecode[
    ```cpp
      int e = static_cast<int>(block.size());
      suffixMaxValues[e - 1] = block[e - 1].totalValue;
      suffixMaxIndices[e - 1] = block[e - 1].index;
      for (int j = e - 2; j >= 0; --j)
      {
          if (block[j].totalValue > suffixMaxValues[j + 1])
          {
              suffixMaxValues[j] = block[j].totalValue;
              suffixMaxIndices[j] = block[j].index;
          }
          else
          {
              suffixMaxValues[j] = suffixMaxValues[j + 1];
              suffixMaxIndices[j] = suffixMaxIndices[j + 1];
          }
      }
    ```
  ],
  caption: "Funzione di calcolo dei suffix max",
  kind: "listing",
)

=== L'algoritmo completo

La funzione completa si articola quindi in questo modo

#figure(
  sourcecode[
    ```cpp
      using namespace std::views;
      auto list = zip(weights, values);
      int n = static_cast<int>(weights.size());
      auto Alist = take(list, n / 2);
      auto Blist = drop(list, n / 2);
      std::vector<CopaSubset> A, B;
      generate_copa_subsets(Alist, numThreads);
      generate_copa_subsets(Blist, numThreads, true);
      int N = static_cast<int>(B.size());
      // Stage 2 : Parallel suffix max for B (MaxBj and Lj)
      std::vector<CopaBlock> blocksA(numThreads);
      std::vector<CopaBlock> blocksB(numThreads);
      distribute_block_per_processor(A, blocksA, numThreads);
      distribute_block_per_processor(B, blocksB, numThreads);
      // Stage 3: Parallel pruning
      std::vector<std::pair<CopaBlock, CopaBlock>> remainingPairs;
      prune(blocksA, blocksB, remainingPairs, capacity, numThreads);
      // Stages 4+5: For each remaining block pair, compute suffix max and search
      int k = static_cast<int>(remainingPairs.size());
      int bestAIdx = 0, bestBIdx = 0, bestValue = 0;
      std::vector<int> localBestVal(k, 0);
      std::vector<int> localBestAIdx(k, 0);
      std::vector<int> localBestBIdx(k, 0);
      parallel_save_max(remainingPairs, localBestVal, localBestAIdx, localBestBIdx, capacity, k);
      // Reduce across all remaining pairs
      #pragma omp parallel for num_threads(numThreads)
      for (int i = 0; i < k; ++i)
      {
          #pragma omp critical
          if (localBestVal[i] > bestValue)
          {
              bestValue = localBestVal[i];
              bestAIdx = localBestAIdx[i];
              bestBIdx = localBestBIdx[i];
          }
      }
      // Reconstruct solution
      KnapsackSolution solution{};
      solution.totalValue = bestValue;
      solution.totalWeight = A[bestAIdx].totalWeight + B[bestBIdx].totalWeight;
      auto items = (A[bestAIdx] << B[bestBIdx]).getItemIndices();
      solution.items.reserve(items.size());
      solution.items.insert(solution.items.end(), items.begin(), items.end());
      return solution;
    ```
  ],
  caption: "Funzione knapsackcopa completa",
  kind: "listing",
)

Questa funzione mette semplicemente insieme tutti i passi dell'algoritmo: 1) si generano i 2 subsets dividendo in 2 la lista di pesi e profitti 2) si distribuiscono i blocchi tra i processori e si cercano i primi massimi profitti tra i blocchi 3) si esegue la potatura dei blocchi 4) si effettua il secondo algoritmo di ricerca dei massimi e si estrae la soluzione finale concatenando gli indici degli oggetti rappresentati come stringa binaria.

== Knapsack COPA MPI

Questo algoritmo è stato concepito principalmente per un utilizzo in shared memory, non è stato pensato per essere utilizzato invece in memoria distribuita. Tuttavia è possibile farne utilizzo, introducendo delle accortezze per minimizzare il più possibile lo scambio di dati. Partendo dalle funzioni di generazione dei subsets

=== Generazione dei subsets

#figure(
  sourcecode[
    ```cpp
      auto proc = compute_corank_procedure(comm, A, B);
      int rank = comm.rank();
      int a_start = proc.a_displacements[rank];
      int a_end = a_start + proc.a_sizes[rank];
      int b_start = proc.b_displacements[rank];
      int b_end = b_start + proc.b_sizes[rank];
      std::vector<CopaSubset> local_result(proc.output_sizes[rank]);
      std::merge(A.begin() + a_start, A.begin() + a_end, B.begin() + b_start, B.begin() + b_end, local_result.begin());
      all_gatherv(comm, local_result.data(), proc.output_sizes[rank], output.data(), proc.output_sizes,
              proc.output_displacements, 0);
    ```
  ],
  caption: "Funzione di merge parallelo della lista",
  kind: "listing",
)

La funzione di merging funziona calcolando parallelamente tutti i corank degli elementi, infine applica il merge su pezzi di lista distinti, esegue il gather delle varie liste usando come displacement proprio gli indici calcolati dal corank e infine distribuisce il risultato a tutti i task.

#figure(
  sourcecode[
    ```cpp
      std::vector<CopaSubset> subsets{CopaSubset{}};
      for (int item_idx = 0; const auto &item : items)
      {
          decltype(subsets) shifted{subsets.size()};
          auto [w, v] = item;
          int local_size = static_cast<int>(shifted.size()) / comm.size();
          int remainder = static_cast<int>(shifted.size()) % comm.size();
          auto procedure = compute_scatter_procedure(comm, subsets);
          std::vector<CopaSubset> local_shifted{static_cast<size_t>(procedure.sizes[comm.rank()]), CopaSubset{}};

          // Distribute subsets to all processes for local shifting
          scatterv(comm, subsets.data(), procedure.sizes, procedure.displacements, local_shifted.data(),
                      procedure.sizes[comm.rank()], 0);

          for (int i = 0; i < static_cast<int>(local_shifted.size()); i++)
          {
              local_shifted[i].addItem(item_idx, w, v);
          }
          all_gatherv(comm, local_shifted.data(), procedure.sizes[comm.rank()], shifted.data(), procedure.sizes,
                  procedure.displacements, 0);
          item_idx++;
          std::vector<CopaSubset> newSubsets;
          newSubsets.resize(subsets.size() + shifted.size());
          mpi_parallel_merge(comm, subsets, shifted, newSubsets);
          subsets = std::move(newSubsets);
      }
      if (reverse)
          std::reverse(subsets);
      return subsets;
    ```
  ],
  caption: "Funzione di generazione dei subsets",
  kind: "listing",
)

La funzione di generazione funziona in modo analogo alla sua controparte shared memory, si crea una lista shifted a partire da quella presente al passo precedente, si esegue lo scattering e ogni task aggiunge internamente al suo pezzo di lista l'oggetto, infine si esegue riunisce la lista e si procede alla funzione di merging parallelo.

=== Distribuzione dei blocchi ai processori

Anche in questo caso la funzione di distribuzione si comporta in modo analogo alla versione shared memory, ogni processore prende un subrange della lista di blocchi e lo salva cercando anche l'elemento di massimo profitto. La funzione non ritorna la lista ma la usa solo per cercare il massimo.

#figure(
  sourcecode[
    ```cpp
      using value_type = std::ranges::range_value_t<decltype(input)>;
      // calculate the size of the blocks for the thread
      int n = static_cast<int>(std::ranges::size(input));
      int k = comm.size();
      int i = comm.rank();

      int block_size = n / k;
      int remainder = n % k;

      int start = i * block_size + std::min(i, remainder);
      int end = start + block_size + (i < remainder ? 1 : 0);
      auto span = std::ranges::subrange(std::ranges::begin(input) + start, std::ranges::begin(input) + end);
      int max_val = 0;
      for (const auto &elem : span)
      {
          if (elem.totalValue > max_val)
              max_val = elem.totalValue;
      }
      return {start, end, max_val};
    ```
  ],
  caption: "Funzione di distribuzione dei blocchi",
  kind: "listing",
)

Dopodichè i processi eseguono il pruning esattamente come nella versione shared memory.

=== Funzione completa

L'aspetto della funzione di calcolo completa è analogo alla versione shared memory, la lista di pesi e profitti viene divisa a metà in 2 liste A e B. Vengono generati i subset per ognuna delle due liste e viene invertito l'ordine per la seconda. Si distribuiscono i blocchi tra i processori e poi ogni processo effettua il pruning comparando i blocchi ad esso assegnati. Finito questo insieme di operazione ogni processo avrà trovato un suo massimo, a quel punto si effettua una reduce su tutte le soluzioni trovate e si prende quella di massimo profitto.

#figure(
  sourcecode[
    ```cpp
      int world = comm.size();
      int n = static_cast<int>(weights.size());
      std::vector<std::pair<int, int>> Alist, Blist;
      Alist.reserve(n / 2);
      Blist.reserve(n / 2);
      for (int i = 0; i < n / 2; ++i)
          Alist.emplace_back(weights[i], values[i]);
      for (int i = n / 2; i < n; ++i)
          Blist.emplace_back(weights[i], values[i]);
      auto A = mpi_generate_copa_subsets(comm, Alist);
      auto B = mpi_generate_copa_subsets(comm, Blist, true);
      int N = static_cast<int>(B.size());
      auto i = comm.rank();
      // Stage 2 : Parallel suffix max for B (MaxBj and Lj)
      CopaBlock blockA;
      std::vector<CopaBlock> blocksB(world);
      std::vector<CopaDistributionIndex> blockBdescriptors(world);
      auto blockAdesc = mpi_distribute_block_per_process(comm, A);
      auto blockBdesc = mpi_distribute_block_per_process(comm, B);
      boost::mpi::all_gather(comm, blockBdesc, blockBdescriptors.data());
      blockA.block = std::span(A).subspan(blockAdesc.start, blockAdesc.end - blockAdesc.start);
      blockA.maxValue = blockAdesc.maxValue;
      for (int i = 0; i < world; ++i)
      {
          blocksB[i].block =
              std::span(B).subspan(blockBdescriptors[i].start, blockBdescriptors[i].end - blockBdescriptors[i].start);
          blocksB[i].maxValue = blockBdescriptors[i].maxValue;
      }
      // Note: we avoid communication during distribution, each node will only communicate the tasks to perform
      std::vector<std::pair<CopaBlock, CopaBlock>> local_remaining_pairs{};
      comm.barrier();
      prune_block_pair(blockA, blocksB, local_remaining_pairs, capacity, i);
      BlockPairSearchResult solution{};
      for (const auto &[remBlockA, remBlockB] : local_remaining_pairs)
      {
          auto local_solution = block_pair_pointer_search(remBlockA, remBlockB, capacity);
          if (local_solution.bestVal > solution.bestVal)
          {
              solution = local_solution;
          }
      }
      BlockPairSearchResult best{};
      reduce(comm, solution,best,boost::mpi::maximum<BlockPairSearchResult>(),0);
      if (comm.rank() == 0)
      {
          // Reconstruct solution
          KnapsackSolution solution{};
          solution.totalValue = best.bestVal;
          solution.totalWeight = A[best.bestAIdx].totalWeight + B[best.bestBIdx].totalWeight;
          //auto items = (A[best.bestAIdx] << B[best.bestBIdx]).getItemIndices();
          solution.items.reserve(items.size());
          solution.items.insert(solution.items.end(), items.begin(), items.end());
          return solution;
      }
      return {};
    ```
  ],
  caption: "Funzione completa COPA MPI",
  kind: "listing",
)

=== Considerazioni finali sull'implementazione

L'implementazione è molto sperimentale e quindi non è sicuramente ottima, ci sono diversi approcci che si possono valutare per esprimere al meglio le capacità dell'algoritmo. Il primo è sicuramente quello di usare una forma più ibrida di calcolo distribuito, in cui ogni task è una farm di thread che collaborano tra di loro in shared memory. Per rendere gli algoritmi più facilmente utilizzabili è stato creato un wrapper python mediante Boost.python, che offre la possibilità di creare classi e funzioni leggibili e eseguibili direttamente dall'interprete, la misura del tempo viene effettuata mediante il modulo time di python, che espone sostanzialmente uno steady clock, azionando l'orologia prima di invocare la funzione C++ e subito dopo alla sua conclusione, per evitare problemi con l'instanza del comunicatore MPI si delega il requisito all'eseguibile python che lo instanzia quando si include la libreria Mpi4py.

=== Strumenti e opzioni di compilazione

La libreria è stata compilata con il compilatore GCC 15.2.0 usando il set di impostazioni di default Release presente in CMake 4.2.0 (quindi -O2 e -DNDEBUG a cui si aggiungono per default -ftree-vectorize e -ffast-math), lo standard usato è C++23 con libstdc++11, l'interprete python è python 3.12.13
