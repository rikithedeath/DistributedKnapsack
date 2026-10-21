=== Implementazione MPI

L'implementazione MPI dell'algoritmo è molto simile, l'approccio seguito è stato quello di assegnare la tabella a un master, che scompone la linea successiva da calcolare in task di equa lunghezza da distribuire ai vari worker.

#figure(
  sourcecode[
    ```cpp
      using namespace boost::mpi;
      int n = static_cast<int>(weights.size());
      std::vector<uint32_t> line(capacity + 1, 0);
      for (int i = 1; i <= n; ++i)
      {
          broadcast(comm, line.data(), static_cast<int>(line.size()), 0);
          while (true)
          {
              NodeTask task{};
              auto status = comm.recv(any_source, any_tag, task);
              if (status.tag() == NODETAG::TERMINATE)
              {
                  break;
              }
              int len = task.endIndex - task.startIndex + 1;
              std::vector<uint32_t> result(len);
              for (int w = task.startIndex; w <= task.endIndex; ++w)
              {
                  if (weights[i - 1] <= w)
                      result[w - task.startIndex] = std::max(line[w], line[w - weights[i - 1]] + values[i - 1]);
                  else
                      result[w - task.startIndex] = line[w];
              }

              comm.send(status.source(), NODETAG::RESPONSE, NodeResponse{task.startIndex, task.endIndex});
              comm.send(status.source(), NODETAG::DATA, NodeResponseData{std::move(result)});
          }
          comm.barrier();
      }
    ```
  ],
  caption: "Worker task",
  kind: "listing",
)

i task sono dei range di indici che il worker deve computare, una volta finito il worker invia al master il blocco appena computato dopodichè il worker attende tutti i nodi alla barriera e si comincia il calcolo per il prossimo oggetto.

#figure(
  sourcecode[
    ```cpp
      int n = static_cast<int>(weights.size());
      std::vector<std::vector<uint32_t>> dp(n + 1, std::vector<uint32_t>(capacity + 1, 0));
      for (int i = 1; i <= n; ++i)
      {
          broadcast(MPI_COMM_WORLD,dp[i - 1].data(),capacity+1,0); //broadcast the previous line from master
          int actual_index = 0;
          int active = 0;
          auto generator = [capacity, chunk_size,&actual_index,&active]() -> std::optional<NodeTask> {
              if (actual_index > capacity)
              {
                  return std::nullopt;
              }
              int start = actual_index;
              int end = std::min(actual_index + chunk_size - 1, capacity);
              actual_index = end + 1;
              active++;
              return NodeTask{start, end};
          };
          send_a_task_to_all_workers(comm, generator);
          while(active>0)
          {
              auto result= receive_a_result_from_any_worker(comm);
              active--;
              copy_result_to_dp_table(dp[i], result);
          }
          terminate_all_workers(comm);
          comm.barrier();
      }
    ```
  ],
  caption: "Master Task",
  kind: "listing",
)

terminati i cicli il master può leggere dalla DP il risultato. Si evince subito che l'implementazione MPI è piuttosto semplice e segue uno schema molto simile alla versione Shared Memory, tuttavia presenta lo svantaggio di dover comunicare la linea calcolata a tutti i worker. Svantaggio che si avrebbe comunque anche nel caso della versione più ottimizzata, in cui non esiste la DP perchè non si esegue il backtracking per ricostruire gli oggetti scelti, ma si calcola solo il profitto massimo ottenibile. In questo caso infatti, per calcolare la linea successiva della DP è necessario conoscere la linea precedente, quindi anche in questo caso si dovrebbe comunicare a tutti i worker la linea precedentemente calcolata.


