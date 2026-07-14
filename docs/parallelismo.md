# Parallelismo nel training RL di LLM

Il ciclo GRPO ha quattro fasi — generazione dei rollout → calcolo reward → forward/backward →
update dei pesi — e ognuna ha una forma di parallelismo naturale diversa.

## 1. Parallelizzare la generazione (copie del modello)

La generazione è la parte "imbarazzantemente parallela": ogni completion è indipendente dalle altre.
In ordine di complessità:

- **Batching su una copia sola**: la stessa forward genera N sequenze insieme; i pesi si leggono
  dalla RAM una volta per tutte le sequenze. È quello che fa `single_proc_launcher.sh`.

- **Data parallel: K copie del modello, prompt divisi in K fette**: ogni copia genera la sua fetta,
  poi un `gather` raccoglie tutte le completion. Con `torchrun` TRL divide i prompt tra i rank e a
  fine generazione fa `gather_object`. Durante la generazione non serve sincronizzare nulla
  (i pesi non cambiano), solo il gather finale. Problema: **sbilanciamento** — le completion hanno
  lunghezze diverse e tutti aspettano il rank con quella più lunga.

- **Server di inferenza dedicato (vLLM)**: il trainer manda i prompt a un processo separato
  ottimizzato solo per generare (continuous batching, KV cache paginata), spesso su GPU diverse
  da quelle di training. In TRL: `use_vllm=True`, modalità *colocate* (vLLM dentro il processo di
  training) o *server* (processo separato). Costo nascosto: a ogni update bisogna ricopiare i pesi
  aggiornati nel server.

## 2. Parallelizzare il backward (data parallelism / DDP)

Vincolo diverso dalla generazione: le K copie devono restare **identiche** dopo ogni update.
Schema DDP (DistributedDataParallel):

1. ogni replica fa forward+backward sulla propria fetta del batch → gradienti *diversi*;
2. un **all-reduce** media i gradienti tra le repliche (ring all-reduce: ogni nodo scambia solo
   con i vicini, banda ottimale);
3. ogni replica applica l'optimizer sulla media → i pesi restano sincronizzati senza mai trasferirli.

Su GPU è efficiente perché i gradienti si all-reducono **a bucket mentre il backward è ancora in
corso**: la comunicazione dei layer finali si sovrappone al calcolo dei layer iniziali.
Su CPU il canale (gloo su loopback) è lento: per Qwen2-0.5B in fp32 sono ~2 GB di gradienti
da mediare a ogni step, e non c'è overlap che tenga.

## 3. Quando il modello non entra (o una copia intera è troppo lenta)

Forme "model parallel", ortogonali al data parallel e combinabili tra loro:

- **Tensor parallelism**: si spezza la *singola matrice* — ogni device tiene una fetta di
  colonne/righe di ogni layer e calcola la sua parte del matmul. Serve un all-reduce delle
  attivazioni *per ogni layer* → vuole interconnessioni velocissime (NVLink), tipicamente entro
  un singolo nodo.

- **Pipeline parallelism**: si spezzano i *layer* — device 1 tiene i layer 1–12, device 2 i 13–24,
  ecc. Il batch si divide in microbatch che scorrono in pipeline; il costo è la "bolla"
  (device fermi all'inizio e alla fine di ogni step).

- **FSDP / ZeRO**: ogni device tiene solo **1/K di parametri, gradienti e stati Adam**; alla
  forward di un layer, un all-gather ricostruisce temporaneamente i pesi completi di quel layer,
  li usa e li scarta. Più comunicazione, ma la memoria scala con K.

## 4. Parallelismo dentro il singolo processo (intra-op)

Le librerie BLAS (oneDNN su CPU) spartiscono le righe di ogni matmul tra i thread
(`torch.set_num_threads` / `OMP_NUM_THREADS`). Zero comunicazione e una sola copia dei pesi,
ma scala solo finché il matmul è abbastanza grande da tenere occupati i core, e mai oltre la
banda RAM.

## 5. La dimensione specifica dell'RL: sovrapporre generazione e training

Nel GRPO sincrono le fasi si alternano: mentre generi non alleni e viceversa. Nei sistemi RL
moderni le due fasi diventano **asincrone**: un pool di worker genera continuamente rollout con
pesi leggermente vecchi mentre il trainer allena su quelli già pronti. Il prezzo: i dati diventano
*off-policy* (policy di 1–2 update fa), corretto con l'importance sampling — che GRPO ha già nella
loss (il ratio clippato con `epsilon` serve esattamente a tollerare questo scarto).
`num_iterations > 1` è una piccola forma della stessa idea: riusa le stesse generazioni per più
update, ammortizzando il costo di generazione.

# Più processi vs più thread (`num_threads`)

Stesso hardware, due modi opposti di usarlo:

|                       | K processi (torchrun/DDP)            | K thread (`num_threads=K`)          |
|-----------------------|---------------------------------------|-------------------------------------|
| memoria               | K spazi separati                      | uno spazio condiviso                |
| copie dei pesi        | K (una per processo)                  | 1                                   |
| copie di grad + Adam  | K                                     | 1                                   |
| lettura pesi per step | K volte dalla RAM                     | 1 volta, ammortizzata sul batch     |
| sincronizzazione      | all-reduce esplicito dei gradienti    | nessuna (i thread scrivono le stesse strutture) |
| granularità           | tra esempi del batch (data parallel)  | dentro il singolo matmul (intra-op) |
| scala su più macchine | sì                                    | no                                  |

- I **processi** hanno spazi di indirizzi separati: ogni copia dei pesi è fisicamente duplicata,
  quindi anche nella L3 condivisa finiscono righe duplicate, e ogni forward ristreaming-a l'intero
  modello dalla RAM. I gradienti vanno mediati esplicitamente (all-reduce).
- I **thread** condividono tutto: la riga di cache caricata da un thread serve anche gli altri,
  i gradienti finiscono direttamente negli stessi tensori, niente comunicazione. Il limite è che
  parallelizzano solo *dentro* le singole operazioni: se il matmul è piccolo, i thread restano
  affamati. (Il GIL di Python non c'entra: i kernel torch lo rilasciano.)
- **Regola pratica**: `num_processi × num_threads ≤ core fisici`, altrimenti oversubscription
  (thread che si contendono i core e si sfrattano a vicenda dalle cache).
  `OMP_NUM_THREADS` va settato *prima* dell'import di torch; `torch.set_num_threads` vale per
  il processo corrente.

**Misurato su questa macchina** (Qwen2-0.5B, effective batch 8, core 0–7): 8 processi × 1 thread
fa step ~2× più lenti di 1 processo × 8 thread (~26–37 s/it contro ~13–17 s/it). Dominano
l'all-reduce dei gradienti e la lettura duplicata dei pesi, non il calcolo.

# Note sulla macchina attuale

**AMD Ryzen Threadripper PRO 9965WX** (Zen 5) — 24 core fisici / 48 thread SMT, 1 socket,
1 nodo NUMA, **502 GB RAM**.

- **Cache**: L1d ~48 KB e L2 1 MB privati per core; L3 128 MB totali ma divisi in **4 slice da
  32 MB**, una per gruppo di 6 core (CCD): core 0–5 → slice 0, 6–11 → slice 1, ecc.
  Attenzione: `taskset -c 0-7` è a cavallo di **due** CCD; per stare dentro una sola L3 usare
  6 core (`0-5`). I CPU 24–47 sono i sibling SMT dei core 0–23 (cpu N ↔ cpu N+24).
- **SMT**: per lavoro BLAS compute/bandwidth-bound i sibling SMT aiutano poco o nulla —
  meglio 1 thread per core fisico (quindi max ~24 thread utili, non 48).
- **AVX-512 completo, incluso `avx512_bf16`**: oneDNN può usare il fast-path bf16 su CPU
  (`torch.autocast("cpu", dtype=torch.bfloat16)`) — dimezza i byte che passano dalla RAM,
  che è proprio il collo di bottiglia; vale la pena provarlo.
- **RAM abbondante (502 GB)**: lo spazio non è mai il problema con un modello da 0.5B
  (ci starebbero decine di copie) — il limite è la **banda**, non la capacità. Piattaforma
  WRX90 a 8 canali DDR5: la banda aggregata è alta, ma resta condivisa tra tutti i core,
  e 8 processi che streammano 8 copie dei pesi la saturano comunque.
- I launcher attuali usano solo 8 dei 24 core: c'è margine per alzare `num_threads`
  (fino a ~24, restando su core fisici) — il guadagno però cala man mano che ci si avvicina
  al limite di banda, e con matmul piccoli (batch piccoli, modello 0.5B) i thread in più
  rendono sempre meno.
