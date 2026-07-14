# RT-GRPO: GRPO con riutilizzo di rollout passati

GRPO standard, dopo ogni update, scarta il generation batch appena usato: ogni rollout viene generato,
consumato per `num_iterations` epoche, e mai più riutilizzato. RT-GRPO estende GRPO tenendo in memoria
gli ultimi rollout (chiamati qui *generation event*, vedi terminologia in `grpo_algo.md`) e mescolandoli
a quello corrente prima di ogni update, per ottenere più segnale di gradiente a parità di costo di
generazione — stessa idea di `RT_PPO`/`MultiRolloutBuffer` applicata a GRPO.

> Implementato in `algorithms/rt_grpo.py` (`RT_GRPOTrainer`) / `buffers/multi_generation_buffer.py`
> (`MultiGenerationBuffer`). Entrambe le correzioni IS (naive e bh) sono attive.

## Perché l'advantage non "invecchia" come in PPO

In PPO l'advantage dipende da una value function che cambia ad ogni update: un vantaggio calcolato
su un rollout vecchio è quindi stimato con una baseline stantia (da qui `fresh_adv`/V-TRACE in `rt_ppo`).

In GRPO l'advantage è `(reward - mean_gruppo) / std_gruppo`: il reward di una risposta già generata
è fisso (non dipende dal modello che si sta allenando) e la normalizzazione è per-gruppo. Riusare un
rollout vecchio **non invalida il suo advantage**. L'unica cosa che invecchia riusando dati vecchi è il
**ratio di importance sampling** (quanto la policy corrente si è allontanata dalla policy che ha generato
quella completion) — RT-GRPO si occupa solo di questo.

## Terminologia aggiuntiva

- **generation event**: l'insieme dei `steps_per_generation` micro-batch prodotti da una chiamata a
  `_generate_and_score_completions` (vedi `grpo_algo.md`). È l'unità della finestra RT — per analogia,
  in `RT_PPO` l'unità è un `collect_rollouts()`.
- **window_length (W)**: generation event totali usati per un update, corrente incluso. `W=0` istanzia
  `GRPOTrainer` standard di TRL; `W=1` istanzia `RT_GRPOTrainer` ma senza history — stesso identico
  algoritmo di `W=0`, solo passando dalla nostra classe invece che direttamente da quella di TRL.
- **window_id**: età di una riga del batch misto, `0` = generation event corrente, `1..W-1` = eventi
  precedenti in ordine di recency.
- **on_policy_mask**: `1.0` se `window_id==0`, altrimenti `0.0`.

## Il buffer (`MultiGenerationBuffer`)

Non è un buffer stile SB3 (`get()`/`add()` a generatore): TRL non ha un punto di iniezione equivalente
a `rollout_buffer_class`, quindi è un oggetto standalone usato internamente da `RT_GRPOTrainer`, con due
soli metodi chiamati da `_prepare_inputs`.

- **`GenerationSnapshot`**: uno per generation event, contiene `slices` — la lista dei suoi
  `steps_per_generation` micro-batch (gli stessi prodotti da `split_tensor_dict`), spostati su CPU, più
  `policy_snapshot` (solo se `is_weight_type="bh"`, vedi sotto). Salvare a granularità di micro-batch,
  non l'evento intero come blob unico, è ciò che permette a `mix()` di riaccostare il micro-batch storico
  giusto a quello corrente per ogni step, senza dover ritagliare nulla a runtime.
- **`history`**: `deque(maxlen=W)`, FIFO puro — `push_new_generation` fa `appendleft` al detection di
  ogni nuovo evento, **prima** di qualsiasi `mix()`. Per contratto, quindi, `history[0]` è sempre
  l'evento corrente, e `mix()` lo salta mescolando la fetta corrente solo con `history[1:]` (i W-1
  eventi precedenti). Quando la deque è piena l'evento più vecchio esce automaticamente; inoltre la
  `policy_snapshot` dell'entry più vecchia *rimasta* viene azzerata subito (non serve più a nessuna
  valutazione bh futura), così in RAM restano al massimo W-1 copie congelate del modello.

```
push_new_generation(slices, policy_snapshot=None):   # chiamato una sola volta per generation event
    history.appendleft(GenerationSnapshot(slices spostate su CPU, policy_snapshot))
    se history è piena: history[-1].policy_snapshot = None   # dati ancora usati, policy mai più


mix(current_slice, step_in_event):             # chiamato una volta per ogni step da _prepare_inputs

    storici = [ evento.slices[step_in_event] for evento in history[1:] ]   # history[0] = evento corrente, saltato
    # ogni evento storico ha esattamente steps_per_generation micro-batch (stesso split nativo
    # subito quando era "corrente"): step_in_event indicizza in modo coerente sia il micro-batch
    # corrente che quelli storici — è quello che tiene allineata la finestra evento per evento

    merged = pad_and_concat([current_slice] + storici)
    # left-pad su prompt_ids/prompt_mask, right-pad su completion_ids/completion_mask/
    # old_per_token_logps/all_log_probs, fino alla lunghezza massima tra tutti i micro-batch inclusi

    merged.window_id      = [0]*len(current_slice) + [1]*len(storici[0]) + [2]*...
    merged.on_policy_mask = (window_id == 0)
    merged.num_items_in_batch = merged.completion_mask.sum()
    # ricalcolato sui token DAVVERO presenti in questo batch misto — sommare i num_items_in_batch
    # "per intero evento" dei micro-batch storici sovrastimerebbe: quel valore conta tutti i loro
    # steps_per_generation micro-batch, non solo quello incluso qui

    ritorna merged
```

`old_per_token_logps` può mancare nel micro-batch appena generato (TRL lo lascia `None` quando generazione
e gradient step sono allineati, vedi `grpo_algo.md`). Prima di `push_new_generation`, `_ensure_old_logps`
lo calcola esplicitamente se assente: senza, una entry storica non avrebbe modo di sapere sotto quale
policy è stata generata quando viene riusata più avanti.

## Il trainer: cosa cambia rispetto a TRL

Lo pseudocode sotto riprende **esattamente** quello di `grpo_algo.md` (sezione "Implementazione in TRL")
e ci sovrappone, evidenziate con `+`, le modifiche e le funzioni aggiunte da `RT_GRPOTrainer`. Le uniche
righe rimosse (`-`) sono quelle sostituite direttamente da una versione RT-equivalente.

Due decisioni di fondo, prima del codice:
- L'aggancio è in `_prepare_inputs`, **dopo** lo split nativo di TRL in micro-batch (`split_tensor_dict`),
  non dentro `_generate_and_score_completions` prima dello split: mescolare prima romperebbe l'assunzione
  di TRL che ogni micro-batch abbia dimensione uguale, e impedirebbe di assegnare `window_id` in modo pulito.
- Il rilevamento di "nuovo generation event" è un confronto per identità su `self._buffered_inputs`
  (non un flag booleano derivato a mano): resta corretto qualunque sia la logica interna con cui TRL
  decide quando rigenerare, senza doverla replicare.

```diff
 Trainer.train()                                        # loop esterno di HF, non modificato

     per ogni generation_batch dal DataLoader:           # GBS prompt, non ancora generati

         training_step(model, generation_batch)
             inputs = _prepare_inputs(generation_batch)
             loss   = compute_loss(model, inputs)         # -> _compute_loss(model, inputs)
             backward(loss)


 _prepare_inputs(generation_batch):                       # <- qui vive il riuso "nativo" (num_iterations/SPG)
+                                                           # RT-GRPO: se window_length <= 1 (o siamo in eval),
+                                                           # nessuna modifica, torna il comportamento invariato

     generate_every = steps_per_generation * num_iterations

     se step % generate_every == 0 (o buffer vuoto):

         batch = _generate_and_score_completions(generation_batch)   # ROLLOUT + REWARD/ADVANTAGE (punti 1 e 2)
         batch = shuffle(batch)
         buffered_inputs = split_tensor_dict(batch, steps_per_generation)   # SPG micro-batch di size uguale

-    ritorna buffered_inputs[step % steps_per_generation]             # il micro-batch di turno, non rigenera
+    inputs = buffered_inputs[step % steps_per_generation]             # il micro-batch di turno, non rigenera
+
+    se buffered_inputs è appena stato rimpiazzato (nuovo generation event, identity check):
+        _ensure_old_logps(buffered_inputs)                # calcola old_per_token_logps se TRL l'ha lasciato None
+        policy_snapshot = None
+        se is_weight_type == "bh":
+            policy_snapshot = _update_all_log_probs_bh(buffered_inputs)         # vedi sotto
+        gen_buffer.push_new_generation(buffered_inputs, policy_snapshot)        # FIFO, maxlen=window_length
+
+    inputs = gen_buffer.mix(inputs, step % steps_per_generation)       # vedi sezione Buffer sopra
+
+    ritorna inputs


 _compute_loss(model, inputs):                             # punto 3, per un singolo micro-batch
+                                                            # RT-GRPO: se window_length <= 1 o "window_id" non
+                                                            # è in inputs, nessuna modifica, invariato come sopra

     per_token_logps = forward(model, inputs)
-    ratio = exp(per_token_logps - inputs["old_per_token_logps"])
+    old_per_token_logps, log_ratio, ratio = _compute_ratio(per_token_logps, mask, inputs)   # vedi sotto
     loss_token = -min(ratio * advantage, clip(ratio, 1-epsilon, 1+epsilon_high) * advantage)
     ...
     ritorna loss
+
+    log_window_diagnostics(ratio, mask, inputs["window_id"], ...)
+    # clip fraction / KL approssimata / |ratio-1| / varianza / ESS, per window_id (0=corrente,
+    # 1..W-1=storico). Se is_weight_type == "bh", logga ANCHE le stesse metriche sul ratio naive
+    # "di controllo" (suffisso _bh per distinguerle), per confrontare i due nello stesso run
+
+
+_compute_ratio(per_token_logps, mask, inputs):              # NUOVO — unico punto esteso per naive/bh,
+                                                               # stesso schema di RT_PPO._compute_ratio
+
+    se is_weight_type == "naive":
+        old_per_token_logps = inputs["old_per_token_logps"]     # quello della policy che ha GENERATO
+                                                                    # quella riga (salvato nello snapshot
+                                                                    # del suo generation event)
+
+    se is_weight_type == "bh":
+        se "all_log_probs" in inputs:
+            old_per_token_logps = bh_effective_old_logps(inputs["all_log_probs"])
+            # media in log-space (logsumexp) delle log-prob sotto TUTTE le policy della finestra
+        altrimenti:
+            old_per_token_logps = ...come naive...       # nessuna history disponibile: bh degenera a naive
+
+    log_ratio = per_token_logps - old_per_token_logps
+    ratio = exp(riduci(log_ratio, importance_sampling_level))    # token o sequence, come GRPO nativo
+
+    ritorna (old_per_token_logps, log_ratio, ratio)
+
+
+_update_all_log_probs_bh(fresh_slices):     # NUOVO — chiamato una volta per generation event, solo se bh attivo
+
+    policy_snapshot = deepcopy(self.model).eval()   # congela la policy CHE HA GENERATO fresh_slices
+                                                       # (nessun training step ancora avvenuto su questi dati)
+
+    # gira PRIMA del push: history = solo eventi precedenti, fino a W; l'eventuale W-esimo sta per
+    # essere espulso dal push imminente, quindi entrambi i loop si fermano ai primi W-1 (history[:W-1])
+
+    per ogni micro-batch in fresh_slices:
+        all_lp[:, :, 0] = la sua old_per_token_logps          # già disponibile, nessun forward pass extra
+        per ogni policy storica ancora in finestra (history[:W-1]):
+            all_lp[:, :, i+1] = eval(policy storica, micro-batch)   # un forward pass per policy storica
+
+    per ogni evento in history[:W-1]:                         # aggiornamento RETROATTIVO
+        per ogni suo micro-batch:
+            nuova_colonna = eval(policy_snapshot, micro-batch)     # la policy appena congelata sui dati vecchi
+            all_lp = concat([nuova_colonna, all_lp])[:, :, :W]      # prepend + tronca, la colonna più vecchia esce
+
+    ritorna policy_snapshot   # salvata nella GenerationSnapshot, liberata dal GC quando il FIFO la scarta
```

`all_log_probs` è una matrice `(rows, T, W)` — **per-token**, non per-sample come in `rt_ppo` (GRPO non
ha un'azione scalare, ha un token per posizione). `old_per_token_logps`/`log_ratio` grezzi sono ritornati
da `_compute_ratio` perché servono anche altrove in `_compute_loss` (rispettivamente: fallback nativo di
`off_policy_mask_threshold`, e input di `vespo`); la riduzione token/sequence è fattorizzata a parte
(`_reduce_log_ratio`) e riusata tale e quale per calcolare anche il ratio naive "di controllo" quando si
fa diagnostica in modalità bh — nessuna duplicazione tra i due casi.

**Costo di bh**: `2*(W-1)` forward pass extra per generation event (non per step) più `W-1` copie
congelate del modello tenute in RAM contemporaneamente — per Qwen2-0.5B in float32 questo significa GB,
non KB come in `rt_ppo` (dove la policy è una MLP di poche decine di KB). Un warning all'avvio stampa la
stima di RAM se `is_weight_type=bh`.

**Nota tecnica (bug corretto, 2026-07-14)**: nella prima versione il buffer aveva `maxlen=W-1` e
`mix()` iterava su tutta la history — ma il push avviene *prima* del mix, quindi `history[0]` era
l'evento corrente: ogni batch misto conteneva l'evento corrente **due volte** (window_id 0 e 1), la
finestra reale era più corta di un evento e con W=2 non c'era alcun riuso di dati vecchi (il
meccanismo RT degenerava in GRPO con batch duplicato). Corretto rendendo esplicito il contratto:
`maxlen=W`, `history[0]` = evento corrente per costruzione, `mix()` usa `history[1:]`, e i loop di
`_update_all_log_probs_bh` (che gira pre-push) si fermano ai primi W-1 eventi.

**Nota tecnica (bug corretto)**: le posizioni di puro padding T (introdotte da `mix()`, mai un token
realmente generato) non hanno nessuna colonna valida in `all_log_probs`. `bh_effective_old_logps` le
sostituisce con `0.0` invece di lasciarle `-inf`: altrimenti, moltiplicate per `completion_mask=0` più
avanti nella loss, produrrebbero `NaN` (per via di `0 * inf = NaN` in IEEE 754) e corromperebbero
l'intera loss della riga, non solo quella posizione.

## Parametri di config (`run/configs/conf_grpo.yaml`)

```yaml
window_length: 0        # generation event totali per update (corrente + storici), 0 = GRPOTrainer standard di TRL
is_weight_type: naive    # correzione IS per dati storici: naive/bh, bh richiede W-1 copie del modello in RAM
```

`window_length == 0` istanzia `GRPOTrainer` standard di TRL; `window_length >= 1` istanzia
`RT_GRPOTrainer` (stesso pattern di scelta `MyPPO`/`RT_PPO` in `run_ppo.py` in base a `window_size`,
adattato: qui lo `0` distingue esplicitamente "nessun wrapper" da "wrapper senza history").

## Idea non implementata: retention per informatività invece che per recency

TRL ha un modulo sperimentale (`trl/experimental/grpo_with_replay_buffer`) che non tiene i rollout per
età ma per utilità: scarta i gruppi con varianza di reward zero (risposte tutte giuste o tutte sbagliate,
advantage sempre 0) e li sostituisce con vecchi gruppi ancora informativi, pescati con probabilità
proporzionale a `|advantage| * std_reward`. È un meccanismo diverso da (e complementare a) la finestra
FIFO qui descritta: FIFO isola la variabile "riuso temporale" e mantiene la distribuzione del dataset
inalterata, il replay buffer introduce una prioritizzazione che confonderebbe le due variabili in un
confronto sperimentale pulito. Lasciato come possibile estensione futura, non nella prima versione.
