# Modello

Qwen2-0.5B-Instruct
- Architettura Transformer con SwiGLU activation, attention QKV bias, group query attention, ecc.
- Pretrained su grande quantità di dati, post-trained con supervised finetuning e direct preference optimization.

# GRPO: Group Relative Policy Optimization

Algoritmo RL on-policy per LLM: stima l'advantage confrontando un **gruppo** di completion generate per
lo stesso prompt, senza bisogno di un value model.

## Dataset e task

Ogni esempio è una coppia `(prompt, solution)`. La `solution` è il valore pulito da predirre.

Il `prompt` è una lista di messaggi (`system`/`user`).

Il modello dato il prompt genera una risposta testuale chiamata `answer` (o `completion`)

La reward function estrae la risposta finale dalla `answer`, e la compara con la `solution`.
(convenzione: il modello dovrebbe scrivere la risposta finale in `\boxed{}`)

## Ciclo di training

Terminologia usata:

- **step**: una chiamata a `optimizer.step()` — è quello che contano `max_steps`, `logging_steps`,
  `eval_steps`, `global_step`.

- **micro-batch**: una collezione di `per_device_train_batch_size * num_processes (N_b)` prompt.

- **gradient_accumulation_steps (GAS)**: per ogni mini-batch viene fatto un forward+backward,
  prima di fare `optimizer.step()` vengono accumulati GAS micro-batch

- **steps_per_generation (SPG)**: quanti micro-batch vengono generati in un rollout.
  Di default `= gradient_accumulation_steps` (un rollout = esattamente un update), ma può essere
  impostato a parte (via `generation_batch_size`) e diventare più grande.

- **generation_batch_size (GBS)**: quanti prompt vengono collezionati ad ogni rollout. `generation_batch_size = steps_per_generation * per_device_train_batch_size * num_processes`.


```

definizione del dataset, dato da n_samples (prompt, solution)


fintantochè non colleziono n_samples * num_train_epochs prompt (se max_step = -1)
fintantochè global_steps < max_step (altrimenti)

    1) ROLLOUT
       
       - campiono dal dataset GBS prompt (SPG micro-batch)
       
       - per ogni prompt campiono G completion con una certa temperature/top_p/top_k/repetition_penalty

       - ottengo SPG*N_b*G sequenze (prompt, completion)


    2) CALCOLO REWARD E ADVANTAGE 

       - per ogni completion calcolo le reward pesate e le sommo:
             reward = Σ reward_weights[i] * reward_fns[i](completion, ...)

       - raggruppo le reward per prompt (gruppi di dimensione G)

       - normalizzo le reward (scale_rewards):
             group -> sottraggo media e divido per std calcolate sul gruppo
                      (le G completion dello stesso prompt)
             batch -> media/std calcolate sull'intero batch (tutti i gruppi insieme)
             none  -> nessuna divisione per std, solo sottrazione della media di gruppo

       - advantage_i = (reward_i - mean_group) / std_group

       - l'advantage è unico per l'intera completion e viene assegnato a ogni suo
         token (nessun value model/GAE come in PPO: il "baseline" è la media del gruppo)


    3) CONSUMO DEL BUFFER (PPO-style)

       per iterazione in num_iterations:    

           per ogni micro-batch in generation_batch_size: (steps_per_generation iterazioni)
               
               ratio = π_θ(token|state) / π_θ_old(token|state)   # old logps dal rollout

               loss_token = -min( ratio * advantage, clip(ratio, 1-epsilon, 1+epsilon_high) * advantage )

               se beta != 0: loss_token += beta * KL(π_θ || π_ref)   # default beta=0 -> stile DAPO

               loss = aggregazione di loss_token sui token attivi (loss_type)

               backward                          # accumula il gradiente, non lo azzera

               ogni gradient_accumulation_steps micro-batch -> optimizer.step()   # <- questo è "uno step"
               
```

Di default: 
  
  - gradient_accumulation_steps = steps_per_generation = 1

  - max_step = -1

```

definizione del dataset, dato da n_samples (prompt, solution)


fintantochè non colleziono n_samples * num_train_epochs prompt

    1) ROLLOUT
       
       - campiono dal dataset N_b prompt (un solo micro-batch)
       
       - per ogni prompt campiono G completion con una certa temperature/top_p/top_k/repetition_penalty

       - ottengo N_b*G sequenze (prompt, completion)


    2) CALCOLO REWARD E ADVANTAGE 

       - per ogni completion calcolo le reward pesate e le sommo:
             reward = Σ reward_weights[i] * reward_fns[i](completion, ...)

       - raggruppo le reward per prompt (gruppi di dimensione G)

       - normalizzo le reward (scale_rewards):
             group -> sottraggo media e divido per std calcolate sul gruppo
                      (le G completion dello stesso prompt)
             batch -> media/std calcolate sull'intero batch (tutti i gruppi insieme)
             none  -> nessuna divisione per std, solo sottrazione della media di gruppo

       - advantage_i = (reward_i - mean_group) / std_group

       - l'advantage è unico per l'intera completion e viene assegnato a ogni suo
         token (nessun value model/GAE come in PPO: il "baseline" è la media del gruppo)


    3) CONSUMO DEL BUFFER (PPO-style)

       per iterazione in num_iterations:    

            ratio = π_θ(token|state) / π_θ_old(token|state)   # old logps dal rollout

            loss_token = -min( ratio * advantage, clip(ratio, 1-epsilon, 1+epsilon_high) * advantage )

            se beta != 0: loss_token += beta * KL(π_θ || π_ref)   # default beta=0 -> stile DAPO

            loss = aggregazione di loss_token sui token attivi (loss_type)

            backward                          

            optimizer.step()   # <- questo è "uno step"
            
```

## Implementazione in TRL

Il ciclo sopra è concettuale. A livello di codice, `GRPOTrainer` lo implementa agganciandosi a due hook
del `Trainer` di HuggingFace (`_prepare_inputs` e `compute_loss`), non con un loop scritto a mano:

```
Trainer.train()                                        # loop esterno di HF, non modificato

    per ogni generation_batch dal DataLoader:           # GBS prompt, non ancora generati

        training_step(model, generation_batch)
            inputs = _prepare_inputs(generation_batch)
            loss   = compute_loss(model, inputs)         # -> _compute_loss(model, inputs)
            backward(loss)


_prepare_inputs(generation_batch):                       # <- qui vive il riuso "nativo" (num_iterations/SPG)

    generate_every = steps_per_generation * num_iterations

    se step % generate_every == 0 (o buffer vuoto):

        batch = _generate_and_score_completions(generation_batch)   # ROLLOUT + REWARD/ADVANTAGE (punti 1 e 2)
        batch = shuffle(batch)
        buffered_inputs = split_tensor_dict(batch, steps_per_generation)   # SPG fette di size uguale

    ritorna buffered_inputs[step % steps_per_generation]             # la fetta di turno, non rigenera


_compute_loss(model, inputs):                             # punto 3, per una singola fetta

    per_token_logps = forward(model, inputs)
    ratio = exp(per_token_logps - inputs["old_per_token_logps"])
    loss_token = -min(ratio * advantage, clip(ratio, 1-epsilon, 1+epsilon_high) * advantage)
    ...
    ritorna loss
```

Punti chiave:
- Il DataLoader restituisce batch grandi (GBS prompt): è `_prepare_inputs`, non il DataLoader, a tagliarli
  in SPG fette e a bufferizzarle in `self._buffered_inputs`.
- `_generate_and_score_completions` viene chiamato solo una volta ogni `generate_every` step; le fette
  successive vengono pescate dal buffer senza rigenerare — è il riuso "entro finestra" del punto 3.
- `split_tensor_dict` taglia in fette di dimensione **esattamente uguale**: vincolo rilevante per RT-GRPO
  (`algorithms/rt_grpo.py`), che mescola dati da rollout precedenti — il merge va fatto dopo questo split,
  non prima, altrimenti la divisione in fette uguali si rompe.

