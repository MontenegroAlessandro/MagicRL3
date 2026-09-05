# Diagnostics

Metriche implementate in POSER. I log standard SB3 mantengono nomi, frequenze e asse delle transizioni ambientali.

**Log standard SB3/PPO**

- `rollout/ep_rew_mean`: quanto reward ottiene in media durante il training.
- `rollout/ep_len_mean`: quanto durano in media gli episodi di training.
- `rollout/success_rate`: quanti episodi di training hanno successo, se disponibile.
- `eval/mean_reward`: quanto reward ottiene durante la valutazione.
- `eval/mean_ep_length`: quanto durano gli episodi di valutazione.
- `eval/success_rate`: quanti episodi di valutazione hanno successo, se disponibile.
- `time/fps`: velocità del training in transizioni ambientali al secondo.
- `time/iterations`: quanti cicli di raccolta e training sono stati eseguiti.
- `time/time_elapsed`: tempo trascorso dall’inizio del training.
- `time/total_timesteps`: quante transizioni sono state raccolte.
- `train/learning_rate`: quanto sono grandi i passi di apprendimento impostati.
- `train/entropy_loss`: indica quanto è casuale la policy; meno negativa significa meno entropia.
- `train/policy_gradient_loss`: obiettivo usato per aggiornare la policy; non è il reward.
- `train/value_loss`: quanto sbaglia il critico rispetto ai target usati nel training.
- `train/loss`: loss totale dell’ultimo minibatch.
- `train/approx_kl`: quanto la policy si discosta dalle policy che hanno raccolto i dati.
- `train/clip_fraction`: quanti ratio sono fuori dai limiti di clipping.
- `train/explained_variance`: quanto i valori salvati nel buffer spiegano le variazioni dei target; vicino a uno è meglio.
- `train/std`: quanto sono disperse le azioni della policy gaussiana, se disponibile.
- `train/n_updates`: quante epoche di ottimizzazione sono state eseguite in totale.
- `train/clip_range`: ampiezza base del clipping della policy.
- `train/clip_range_vf`: quanto può cambiare il valore prima del clipping del critico, se attivo.

**Diagnostics POSER**

Convenzione: `diag/nome_metrica/wXX` per ogni window; `/global` solo per una media globale, come la deviazione media dei ratio o la frazione di campioni clippati. Gli altri valori globali non hanno suffisso. Gli assi usano `diag/time/nome`. Le statistiche globali sui ratio si calcolano su tutti i campioni insieme.

Una window è un rollout: `w00` è il più recente. Il ratio confronta la probabilità dell’azione sotto la policy attuale con quella sotto la policy che l’ha raccolta.

- `diag/d2_rad/wXX` per window: radice quadrata di D; cresce quando il rollout è meno compatibile con la policy corrente.
- `diag/d2/wXX` per window: secondo momento analitico D del ratio, mediato sugli stati del rollout; vicino a uno indica buona compatibilità.
- `diag/weight/wXX` per window: peso usato nell’epoca appena conclusa; prima del training o quando PSR blocca, mostra il peso candidato.
- `diag/abs_eps/wXX` e `diag/abs_eps/global` per window e medio: media di `|1 - ratio|`; mostra quanto i ratio si allontanano da uno, non il clip range.
- `diag/clip_range/wXX` per window: ampiezza usata nell’epoca appena conclusa; prima del training o quando PSR blocca, mostra quella candidata.
- `diag/clip_fraction/wXX` e `diag/clip_fraction/global` per window e media: quota di ratio fuori dai limiti effettivi di clipping.
- `diag/ratio_variance/wXX` e `diag/ratio_variance` per window e globale: quanto sono dispersi i ratio; picchi indicano correzioni IS molto disomogenee.
- `diag/clipped_ratio_variance/wXX` e `diag/clipped_ratio_variance`: quanto sono dispersi i ratio dopo il clipping con i limiti effettivi della loss; confrontarla con la varianza prima del clipping.
- `diag/ess_empirical/wXX` normalizzata per window: quanto i ratio osservati concentrano il peso su pochi campioni; vicino a uno significa poca concentrazione.
- `diag/ess_analytic/wXX` normalizzata per window: compatibilità delle distribuzioni delle azioni calcolata tramite D2; vicino a zero segnala rischio anche se i campioni osservati sembrano regolari.
- `diag/time/epoch`: epoche completate nell’update corrente; zero indica la misura prima del training.
- `diag/time/flat_step`: contatore crescente delle misure, per seguire le epoche senza sovrascrivere i punti.
- `diag/time/env_step`: transizioni raccolte al momento della misura, per confrontare run diversi.
- `diag/psr_value`: valore del criterio PSR, calcolato anche con soglia disattivata; blocca nuove epoche solo se il controllo è attivo.
- `diag/psr_triggered`: indica se PSR ha effettivamente bloccato un’epoca, inclusa la prima.
- `diag/critic/normalized_mse/wXX`: errore quadratico del critico diviso per la varianza dei target; confrontarlo prima e dopo l’update sugli stessi target appena ricalcolati, mantenuti fissi. Più basso è meglio; `NaN` se la varianza dei target è minore o uguale a `1e-8`.
- `diag/critic/vtrace_target_change_rms/wXX`: quanto cambiano i target dei rollout storici dopo il ricalcolo V-trace; distingue il movimento dei target dalla difficoltà del critico a impararli.
- `diag/critic/bias/wXX`: media di predizione meno target ricalcolato; positivo indica sovrastima, negativo sottostima. Confrontarlo prima e dopo l’update sugli stessi target fissi.

Qui `d2` indica D e `d2_rad` indica la radice di D: nessuno dei due è normalizzato. Solo `weight` contiene i pesi normalizzati. `abs_eps` corrisponde all’attuale `diag/abs_ratio`.

**Quando vengono registrate**

- Uno snapshot prima dell’update (`epoch=0`) e uno dopo ogni epoca completata, prima del discard.
- Se PSR blocca dopo un’epoca, uno snapshot aggiuntivo con lo stesso `epoch` registra i pesi candidati e `psr_triggered=1`.
- Ogni snapshot ha un `flat_step` crescente, usato come global step delle diagnostics; `env_step` resta fermo durante le epoche dello stesso update.
- Dopo un’epoca, pesi e clipping sono quelli appena usati; `d2`, `d2_rad` ed ESS descrivono invece la policy aggiornata.
- Il cambiamento dei target V-trace è registrato una sola volta, nello snapshot iniziale e solo per i rollout storici.
- MSE e bias usano il critico corrente senza value clipping e gli stessi target mantenuti fissi durante l’update.
- Le varianze dei ratio sono campionarie; le statistiche globali usano tutti i campioni insieme.
- I log PSR sono sempre presenti, anche con soglia disattivata; in quel caso `psr_triggered` resta zero; un superamento misurato dopo l’ultima epoca non conta come arresto.

**W&B**

Nel runner POSER, `wandb.sync_tensorboard=true` abilita l’invio diretto delle metriche a W&B. La sincronizzazione automatica degli eventi TensorBoard viene disattivata per evitare di mescolare i due contatori; i file TensorBoard restano disponibili.

Le diagnostics usano `diag/time/flat_step` come asse; i log SB3 usano `time/total_timesteps`. Ogni dump viene inviato come riga distinta. Con `wandb.sync_tensorboard=false` le metriche non vengono inoltrate. Il ramo MyPPO mantiene la sincronizzazione TensorBoard precedente.
