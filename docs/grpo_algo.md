# GRPO: Group Relative Policy Optimization

Algoritmo RL on-policy per LLM: stima l'advantage confrontando un **gruppo** di
completion generate per lo stesso prompt, senza bisogno di un value model. Questo
documento parte dal vocabolario di base (pensato per chi non conosce già i nomi dei
parametri o il gergo RL/PPO applicato agli LLM) prima di mostrare lo pseudocodice.

## Modello

Qwen2-0.5B-Instruct
- Architettura Transformer con SwiGLU activation, attention QKV bias, group query attention, ecc.
- Pretrained su grande quantità di dati, post-trained con supervised finetuning e direct preference optimization.

## Parametri principali

Nomi esatti come compaiono in `run/configs/conf_grpo.yaml` e nella `GRPOConfig` di
TRL.

### Dati e task

| Parametro | Significato |
|---|---|
| `task_name` | quale dataset/task usare (es. `multiarith`, `gsm8k`) |
| `n_samples` | quanti esempi `(prompt, solution)` usare dal dataset |
| `seed` | seed globale, per riproducibilità (campionamento dataset, generazione, init) |
| `num_train_epochs` | quante volte l'intero dataset viene ripassato |
| `shuffle_dataset` | se rimescolare l'ordine del dataset ad ogni epoca |

### Generazione (rollout)

Per ogni prompt, il modello genera `num_generations` completion campionando token
per token:

| Parametro | Significato |
|---|---|
| `num_generations` | quante completion generare per ogni prompt (dimensione del gruppo) |
| `max_completion_length` | numero massimo di token generabili per completion (oltre viene troncata) |
| `temperature` | quanto casuale è il campionamento (più alta = più varietà tra le completion dello stesso gruppo) |
| `top_p` / `top_k` / `min_p` | filtri sul campionamento (nucleus sampling, top-k, soglia di probabilità minima) |
| `repetition_penalty` | penalizza (o incentiva) la ripetizione di token già usati |

### Dimensioni di batch e scheduling

Questa è la parte più insidiosa in termini di nomi. Un rollout è composto da `generation_batch_size` sequenze. Una sequenza è data da un prompt e una completion. Per collezionare il rollout si possono usare più processi in parallelo. Il microbatch è l'unità su cui si calcola il gradiente.

| Parametro | Significato |
|---|---|
| `per_device_train_batch_size` | quante sequenze (prompt+completion) processa un singolo device in un micro-batch. Contiene `per_device_train_batch_size / num_generations` prompt unici, ciascuno ripetuto `num_generations` volte (una volta per ogni completion del suo gruppo) |
| numero di processi | quanti device (GPU/processi) girano in parallelo |
| `gradient_accumulation_steps` | quanti micro-batch vengono accumulati (forward+backward) prima di un vero update dei pesi (`optimizer.step()`) |
| `steps_per_generation` | quanti micro-batch vengono generati insieme in un unico "rollout" (evento di generazione), prima di generarne di nuovi. Default: uguale a `gradient_accumulation_steps` |
| `generation_batch_size` | quante sequenze totali vengono generate in un rollout: `generation_batch_size = per_device_train_batch_size × numero di processi × steps_per_generation`. Contiene quindi `generation_batch_size / num_generations` prompt unici |
| `num_iterations` | quante volte le stesse completion generate vengono riusate per fare update successivi, prima di generarne di nuove (riuso "on-policy soft": più alto = più update per generazione, ma con dati via via più "vecchi") |

**Training distribuito (più processi)**: `per_device_train_batch_size` è già "per
device" — il micro-batch che un singolo processo elabora in un forward+backward non
si moltiplica per `numero di processi`. Quello che cambia con più processi è la
quantità di dati coperta in parallelo: prima di ogni `optimizer.step()` i gradienti
calcolati sui micro-batch locali di tutti i processi vengono sincronizzati (mediati),
quindi ogni processo applica lo stesso identico update alla propria copia dei pesi.
Il numero di `optimizer.step()` per generation event non dipende da `numero di
processi` (`steps_per_generation × num_iterations / gradient_accumulation_steps`),
ma il numero totale di `optimizer.step()` per finire il dataset diminuisce
all'aumentare di `numero di processi`, perché ogni step copre più dati.

### Reward e advantage

| Parametro | Significato |
|---|---|
| `reward_fns` | lista di funzioni di reward, ciascuna con un peso (`weight`, default 1.0) |
| `reward_weights` | i pesi con cui le reward delle diverse funzioni vengono sommate |
| `scale_rewards` | come si normalizza la reward per calcolare l'advantage — vedi sotto, è l'unico punto realmente sottile |
| `multi_objective_aggregation` | in che ordine si sommano/normalizzano reward multiple (default: prima somma pesata, poi normalizzazione) |

`scale_rewards` ha tre modalità: **la media sottratta è sempre quella del gruppo**,
in tutti e tre i casi. Cambia solo *come si calcola la deviazione standard* usata
per dividere:

- `"group"` (default): std calcolata sulle `num_generations` completion dello
  stesso gruppo.
- `"batch"`: std calcolata su tutte le completion del batch (tutti i gruppi
  insieme) — ma la media resta quella del singolo gruppo, non quella del batch.
- `"none"` / `False`: nessuna divisione per std, solo sottrazione della media di
  gruppo.


### Loss e ottimizzazione

| Parametro | Significato |
|---|---|
| `learning_rate` | passo dell'optimizer (AdamW) |
| `epsilon` | quanto il ratio può scendere sotto 1 prima di essere "tagliato" (clipping inferiore, stile PPO) |
| `epsilon_high` | quanto il ratio può salire sopra 1 prima di essere tagliato (clipping superiore). Se non specificato, usa lo stesso valore di `epsilon` |
| `beta` | peso della penalità KL rispetto al reference model. `0.0` (default in questa repo) = nessun vincolo, stile DAPO — il reference model non viene nemmeno caricato |
| `loss_type` | come si aggregano le loss dei singoli token in un'unica loss scalare (`dapo` è il default usato in questa repo: normalizza sul numero di token attivi nell'intero batch accumulato, elimina un bias legato alla lunghezza delle completion) |
| `mask_truncated_completions` | se escludere dalla loss le completion troncate (superano `max_completion_length` senza terminare) |
| `max_grad_norm` | soglia di gradient clipping, evita update troppo grandi |

## Pseudocodice

Versione concettuale, un ciclo per volta:

```
per ogni rollout (finché non ho fatto n_samples * num_train_epochs prompt, oppure max_steps step):

    1) GENERAZIONE
       - prendo generation_batch_size / num_generations prompt unici dal dataset
         (generation_batch_size = per_device_train_batch_size * numero di processi * steps_per_generation)
       - per ognuno, genero num_generations completion campionando con temperature/top_p/top_k/repetition_penalty
       - ottengo generation_batch_size sequenze (prompt, completion) totali — num_generations copie per ogni prompt

    2) REWARD E ADVANTAGE
       - per ogni completion calcolo:
             reward = Σ reward_weights[i] * reward_fns[i](completion, ...)
       - raggruppo le reward per prompt (gruppi di dimensione num_generations)
       - per ogni completion:
             advantage = reward - media_del_suo_gruppo
             se scale_rewards != "none": advantage /= std(gruppo, oppure batch se scale_rewards="batch")

    3) UPDATE (ripetuto num_iterations volte sulle stesse completion generate, PPO-style)
       per ogni iterazione in num_iterations:
           per steps_per_generation volte (un micro-batch di per_device_train_batch_size sequenze ad ogni iterazione):

               ratio = π_θ(token | stato) / π_θ_old(token | stato)     # π_θ_old = pesi al momento della generazione

               loss_token = -min( ratio * advantage,
                                   clip(ratio, 1-epsilon, 1+epsilon_high) * advantage )

               se beta != 0:
                   loss_token += beta * KL(π_θ || π_ref)

               loss = aggregazione di loss_token sui token attivi (dipende da loss_type)

               backward(loss)                      # accumula il gradiente, non lo azzera

               ogni gradient_accumulation_steps micro-batch → optimizer.step()   # <- questo è un vero "step" di ottimizzazione
```

Nel caso più semplice (quello usato di default in questa repo:
`gradient_accumulation_steps = steps_per_generation = 1`), un rollout produce
esattamente un micro-batch, che viene consumato `num_iterations` volte, e ogni
iterazione corrisponde a un vero `optimizer.step()`:

```
per ogni rollout:

    1) GENERAZIONE: per_device_train_batch_size / num_generations prompt unici,
       num_generations completion ciascuno → per_device_train_batch_size sequenze totali

    2) REWARD E ADVANTAGE: come sopra

    3) per iterazione in num_iterations:
           ratio = π_θ(token|stato) / π_θ_old(token|stato)
           loss_token = -min( ratio*advantage, clip(ratio, 1-epsilon, 1+epsilon_high)*advantage )
           se beta != 0: loss_token += beta * KL(π_θ || π_ref)
           loss = aggregazione(loss_type)
           backward
           optimizer.step()
```

## Implementazione in TRL

Il ciclo sopra è concettuale. A livello di codice, `GRPOTrainer` lo implementa
agganciandosi a due hook del `Trainer` di HuggingFace (`_prepare_inputs` e
`compute_loss`), non con un loop scritto a mano:

```
Trainer.train()                                        # loop esterno di HF, non modificato

    per ogni generation_batch dal DataLoader:           # generation_batch_size sequenze, non ancora generate

        training_step(model, generation_batch)
            inputs = _prepare_inputs(generation_batch)
            loss   = compute_loss(model, inputs)         # -> _compute_loss(model, inputs)
            backward(loss)


_prepare_inputs(generation_batch):                       # <- qui vive il riuso "nativo" (num_iterations/steps_per_generation)

    generate_every = steps_per_generation * num_iterations

    se contatore_interno % generate_every == 0 (o buffer vuoto):

        batch = _generate_and_score_completions(generation_batch)   # ROLLOUT + REWARD/ADVANTAGE (punti 1 e 2)
        batch = shuffle(batch)
        buffered_inputs = split_tensor_dict(batch, steps_per_generation)   # steps_per_generation fette di size uguale

    ritorna buffered_inputs[contatore_interno % steps_per_generation]   # la fetta di turno, non rigenera


_compute_loss(model, inputs):                             # punto 3, per una singola fetta

    per_token_logps = forward(model, inputs)
    ratio = exp(per_token_logps - inputs["old_per_token_logps"])
    loss_token = -min(ratio * advantage, clip(ratio, 1-epsilon, 1+epsilon_high) * advantage)
    ...
    ritorna loss
```

**Attenzione a un dettaglio**: il "contatore_interno" usato in `_prepare_inputs` (nel
codice, `self._step`) **non è lo stesso contatore** che avanza ad ogni vero
`optimizer.step()` (`global_step`, quello che regola `max_steps`/`logging_steps`).
`self._step` si incrementa ad ogni micro-batch, cioè ad ogni chiamata a
`training_step` — prima ancora che l'accumulo di gradiente scatti l'update vero e
proprio. Con `gradient_accumulation_steps=1` (il caso di questa repo) i due
contatori coincidono numericamente, quindi la distinzione non cambia nulla in
pratica qui — ma concettualmente sono due cose diverse, e con
`gradient_accumulation_steps > 1` andrebbero tenute separate.

