#import "../functions/preamble.typ": *

#show: style-algorithm
= Knapsack DP GPU

La versione su GPU (NVIDIA, tramite CUDA) dell'algoritmo dello zaino in versione dynamic programming riprende la frontiera di esecuzione del capitolo classico: la pseudocodifica è infatti la stessa presentata in @algo-kpdp, cambia sostanzialmente la macchina su cui la frontiera viene eseguita. Invece di un insieme di thread della CPU che collaborano su una riga della DP, ogni cella della riga corrente viene assegnata a un thread del grid CUDA; l'accesso agli elementi $w$ della riga è contiguo, quindi i thread di uno stesso warp effettuano accessi alla memoria globale e la ricorrenza non richiede strutture di supporto aggiuntive.

== Knapsack DP GPU: implementazione

Il cuore dell'implementazione è il kernel che calcola una riga della DP, ogni thread calcola esattamente una cella.

#figure(
  caption: "Knapsack DP GPU: kernel del passo di DP",
  kind: "listing",
  sourcecode[
    ```cpp
    __global__ void knapsack_step_kernel(const int *__restrict__ prev, int *__restrict__ cur, int weight, int value,
                                         int cols)
    {
        const int w = blockIdx.x * blockDim.x + threadIdx.x;
        if (w >= cols)
        {
            return;
        }
        const int exclude = prev[w];
        const int include = (weight <= w) ? prev[w - weight] + value : exclude;
        cur[w] = include > exclude ? include : exclude;
    }
    ```
  ],
)



#figure(
  caption: "Knapsack DP GPU: loop host con un lancio kernel per riga",
  kind: "listing",
  sourcecode[
    ```cpp
    const int cols = capacity + 1;
    const std::size_t tableSize = static_cast<std::size_t>(n + 1) * cols;

    const int threads = resolve_threads();
    const int blocks = default_blocks(threads, cols);
    int *d_dp = nullptr;

    cuda_check(cudaMalloc(&d_dp, tableSize * sizeof(int)), "cudaMalloc");
    // Only row 0 must be zero (dp[0][w] = 0); rows 1..n are fully overwritten.
    cuda_check(cudaMemset(d_dp, 0, static_cast<std::size_t>(cols) * sizeof(int)), "cudaMemset");
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    const auto dpStart = std::chrono::steady_clock::now();

    for (int i = 1; i <= n; ++i)
    {
        const int *prev = d_dp + static_cast<std::size_t>(i - 1) * cols;
        int *cur = d_dp + static_cast<std::size_t>(i) * cols;
        knapsack_step_kernel<<<blocks, threads>>>(prev, cur, weights[i - 1], values[i - 1], cols);
        cuda_check(cudaGetLastError(), "knapsack_step_kernel");
    }
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    const auto dpEnd = std::chrono::steady_clock::now();
    const double dpMs = std::chrono::duration<double, std::milli>(dpEnd - dpStart).count();
    g_last_dp_time_ms = dpMs;
    std::printf("GPU DP execution: %.3f ms\n", dpMs);
    ```
  ],
)

Il kernel è la traduzione più diretta possibile della frontiera classica,  la riga precedente viene letta direttamente dalla memoria globale. Questo rende l'algoritmo, come nella versione CPU, memory bound per costruzione: per ogni cella si leggono due valori e se ne scrive uno, mentre la parte aritmetica si riduce a una comparazione e una addizione.

La gestione della dipendenza fra righe è il punto in cui il modello CUDA differenza maggiormente da OpenMP: non esistono barriere fra blocchi né buffer ping-pong. La sincronizzazione è garantita implicitamente dal modello di esecuzione degli stream: i kernel consecutivi lanciati sullo stesso stream vengono eseguiti in ordine, quindi il kernel della riga $i$ vede la riga $i-1$ completa prima di partire. La dimensione del blocco è configurable mediante la variabile d'ambiente `KP_GPU_THREADS` (default 256) e viene limitata al massimo supportato dal device (`cudaDevAttrMaxThreadsPerBlock`).



== Knapsack DP GPU: integrazione e build

L'integrazione nel resto del progetto avviene mediante una libreria separata: il target `distributed_knapsack_gpu` viene generato dalla sottodirectory `gpu/` solo quando l'opzione CMake `KP_GPU` è attiva (preset `gpu`, o `all` che combina GPU e MPI), e linkata al binario principale. Dal punto di vista utente l'algoritmo si seleziona con `--test knapsackdp_gpu`; il numero di thread per blocco si regola con la variabile d'ambiente `KP_GPU_THREADS` mentre il tempo di esecuzione del solo kernel DP viene misurato lato host con `std::chrono` ed esposto dalla funzione `kp::gpu::last_dp_time_ms()`, in modo che le eventuali campagne di misura possano separare il tempo del calcolo da quello dei trasferimenti. Gli errori CUDA vengono sempre propagati come eccezioni mediante il wrapper `cuda_check`, che converte lo status in un messaggio leggibile.

== Knapsack DP GPU: Risultati

=== Esecuzione dell'esperimento

La campagna di misure è stata eseguita sul cluster mercurio con lo script Slurm riportato di seguito: per ogni combinazione di famiglia di peso ($"maxWeight" in {10, 1000}$), thread per blocco ($"KP_GPU_THREADS" in {32, 64, 128, 256, 512, 1024}$) e capacity ($10^5$, $10^6$, $10^7$) vengono ripetute 3 misure; per ogni replica lo script raccoglie dallo stdout di `kpbench` sia il tempo di esecuzione (wall time) del solver sia il tempo del solo ciclo di kernel DP, apponendoli a un CSV di riepilogo. La partizione espone una NVIDIA A100-PCIE-40GB (40 GB di memoria HBM2e, banda di picco dichiarata $approx 1.55" TB"\/"s"$). Si noti che la griglia contiene sempre $ceil((C+1)/t)$ blocchi, cioè un thread per cella: il parallelismo totale è fisso, il numero di thread per blocco cambia solo la geometria di lancio e l'occupazione delle SM. Le taglie in $"KP_GPU_THREADS"$ seguono proprio la struttura dell'hardware: i thread di un blocco vengono raggruppati ed emessi in warp di 32 lane, l'unità di schedulazione dell'SM, quindi si considerano solo multipli di 32; una taglia non allineata genererebbe un ultimo warp parzialmente vuoto che consuma comunque uno slot di schedulazione.

#figure(
  caption: "sbatch della campagna di misure sulla versione GPU",
  kind: "listing",
  sourcecode[
    ```bash
    #!/bin/bash
    #SBATCH --job-name=knapsack_gpu
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=8
    #SBATCH --gres=gpu:1
    #SBATCH --time=02:00:00
    #SBATCH --output=gpu.out
    #SBATCH --error=gpu.err
    #SBATCH --mem=32G

    set -euo pipefail

    SEED=42
    NUM_ITEMS=50
    BIN="./build/kpbench"
    REPEATS=3
    TEST="knapsackdp_gpu"
    OUT_DIR="gpu_out"
    SUMMARY="$OUT_DIR/gpu_summary.csv"

    mkdir -p "$OUT_DIR"
    echo "max_weight,capacity,threads,repeat,time_s,dp_ms" > "$SUMMARY"

    for max_weight in 10 1000; do
      for threads in 32 64 128 256 512 1024; do
        for cap in 100000 1000000 10000000; do
          export KP_GPU_THREADS="$threads"
          for rep in $(seq 1 "$REPEATS"); do
            RUN_OUT=$("$BIN" --test "$TEST" --capacity "$cap" \
              --numItems "$NUM_ITEMS" --seed "$SEED" \
              --maxWeight "$max_weight" --save "")
            time_s=$(printf '%s\n' "$RUN_OUT" | \
              sed -n "s/^${TEST}: \([0-9.]*\)s .*/\1/p")
            dp_ms=$(printf '%s\n' "$RUN_OUT" | \
              sed -n "s/^GPU DP execution: \([0-9.]*\) ms/\1/p")
            echo "$max_weight,$cap,$threads,$rep,$time_s,$dp_ms" >> "$SUMMARY"
          done
        done
      done
    done
    ```
  ],
) <lst:kpdpgpu-sbatch>

=== Effetto dei thread per blocco

#figure(
  image("../images/perf_kpdpgpu_block_size.png", width: 100%),
  caption: [Tempo del ciclo di kernel DP (sopra) e rapporto di tempo rispetto alla configurazione a 32 thread per blocco (sotto) #footnote[al variare dei thread per blocco, per famiglia di peso e capacity. Il parallelismo totale è fisso (un thread per cella): il rapporto misura l'effetto della geometria di lancio, non dello scaling.]],
) <fig:kpdpgpu-block-size>

La prima osservazione riguarda il tempo di esecuzione: a prescindere da capacity e block size il solver completa in $0.28 - 0.37" s"$, essenzialmente costante al variare dei parametri. Questo tempo è dominato dal setup CUDA (creazione del contesto, allocazione della tabella fino a 400 MB, trasferimenti e backtrack), che viene pagato una volta per esecuzione; la metrica significativa è quindi il tempo del ciclo di kernel `dp_ms`.

A capacity $10^5$ e $10^6$ il tempo del kernel resta piatto a $approx 3 - 5" ms"$ qualunque sia la taglia: il run è in un regime dominato dall'overhead dei $n$ lanci sequenziali e dalla sincronizzazione. L'effetto del block size emerge solo a capacity $10^7$, dove il lavoro per lancio supera l'overhead: come si vede in @fig:kpdpgpu-block-size, passando da 32 a 256 thread per blocco il tempo scende da $approx 19$ a $approx 6.8" ms"$ ($"maxWeight" = 10$) e da $approx 16$ a $approx 5.8" ms"$ ($"maxWeight" = 1000$), una riduzione di $approx 2.8 times$; oltre 256 thread il tempo risale leggermente.

=== Confronto con la versione CPU

#nota(titolo: "Confronto da rifare")[La versione CPU di riferimento qui è la DP classica *column-major* (@sec:kpdp-layout), che sulla macchina della verifica preliminare è circa 13 volte più lenta di quella row-major: i rapporti GPU/CPU che seguono sovrastimano il vantaggio della GPU e vanno ricalcolati sulle nuove misure CPU.]



Il confronto con la versione CPU classica (stesse istanze, stesso seed) va letto su due piani. Sul tempo di esecuzione la GPU vince nettamente solo a capacity $10^7$: $22 - 23 times$ contro il seriale e $approx 3.2 times$ contro 32 thread. A capacity $10^6$ invece il tempo di esecuzione della GPU ($approx 0.30" s"$, fisso) è *peggiore* del CPU a 32 thread ($approx 0.09" s"$) e appena migliore del seriale ($approx 0.65" s"$): l'overhead di setup (~0.3 s) non è ammortizzato da un kernel che dura pochi millisecondi.

Sul solo tempo di calcolo il quadro cambia di due ordini di grandezza: il kernel GPU impiega $approx 3.6" ms"$ contro i $640" ms"$ del seriale a capacity $10^6$ ($approx 140 - 180 times$) e $approx 6 - 7" ms"$ contro $6.6 - 6.8" s"$ a capacity $10^7$ ($approx 900 times$); anche contro il CPU a 32 thread il vantaggio resta $approx 18 - 24 times$ e $approx 120 - 130 times$ rispettivamente. La conclusione è quella attesa per un workload memory-bound su architetture eterogenee: il kernel GPU satura la banda del device ed è enormemente più veloce nel calcolo puro, ma per istanze singole e con tabelle che stanno in cache l'overhead fisso di setup cancella il vantaggio e la versione GPU diventa dominante quando il problema cresce.
