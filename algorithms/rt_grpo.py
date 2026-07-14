import copy
import itertools
import warnings
from typing import Literal

import torch
import trl
from trl import GRPOTrainer
from trl.trainer.utils import nanmax, nanmin

from buffers.multi_generation_buffer import MultiGenerationBuffer

# Versione di TRL su cui è stato copiato/adattato il corpo di _compute_loss (vedi
# _compute_loss_with_diagnostics). Se .venv aggiorna trl, ri-diffare quel metodo contro
# GRPOTrainer._compute_loss nella nuova versione prima di fidarsi dei risultati.
_PINNED_TRL_VERSION = "1.7.0"

IS_WEIGHT_TYPE = Literal["naive", "bh"]


class RT_GRPOTrainer(GRPOTrainer):
    """
    GRPOTrainer con una finestra scorrevole di generation event passati (vedi docs/rt_grpo_algo.md).

    GRPO standard scarta ogni generation event dopo l'update. Questo trainer tiene i precedenti
    window_length-1 generation event e li rimescola con quello corrente prima di ogni update,
    dando più segnale di gradiente a parità di costo di generazione.

    L'aggancio è in _prepare_inputs, DOPO lo split nativo di TRL in steps_per_generation fette
    (non in _generate_and_score_completions, prima dello split): mescolare prima dello split
    romperebbe l'assunzione di TRL che ogni fetta abbia dimensione uguale.

    window_length <= 1 → identico a GRPOTrainer (nessuna history, nessun overhead).

    :param window_length: generation event totali usati per update (corrente + storici).
    :param is_weight_type: correzione IS per i dati storici.
        "naive": ratio contro la policy che ha generato il campione, via old_per_token_logps — il
        clipping PPO/DAPO nativo di TRL fa il resto. Costo aggiuntivo trascurabile.
        "bh" (balance heuristic): ratio contro la media in log-space delle log-prob sotto tutte
        le policy della finestra (stesso principio di RT_PPO._compute_ratio, adattato al caso
        per-token). Richiede tenere window_length-1 copie congelate del modello in RAM e forward
        pass extra ad ogni generation event — un warning stampa la stima di RAM all'avvio.
    """

    def __init__(self, *args, window_length: int = 1, is_weight_type: IS_WEIGHT_TYPE = "naive", **kwargs):
        if window_length < 1:
            raise ValueError(
                "window_length deve essere >= 1 per RT_GRPOTrainer; per il GRPOTrainer standard di "
                "TRL usa window_length=0 nel wiring di run_grpo.py."
            )

        self.window_length = window_length
        self.is_weight_type = is_weight_type

        super().__init__(*args, **kwargs)

        if trl.__version__ != _PINNED_TRL_VERSION:
            warnings.warn(
                f"RT_GRPOTrainer._compute_loss_with_diagnostics è una copia adattata di "
                f"GRPOTrainer._compute_loss per trl=={_PINNED_TRL_VERSION}, ma è installato "
                f"trl=={trl.__version__}. Ri-diffare il metodo contro il nuovo sorgente TRL prima "
                f"di fidarsi della diagnostica per finestra (la loss nativa non è affetta, solo "
                f"la parte di logging diagnostics_*).",
                stacklevel=2,
            )

        self._gen_buffer = (
            MultiGenerationBuffer(window_length=window_length, pad_token_id=self._tokenizer.pad_token_id)
            if window_length > 1
            else None
        )

        if self.is_weight_type == "bh" and window_length > 1:
            param_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())
            extra_gb = (window_length - 1) * param_bytes / 1e9
            warnings.warn(
                f"is_weight_type='bh' con window_length={window_length}: verranno mantenute "
                f"{window_length - 1} copie congelate del modello in RAM (~{extra_gb:.1f} GB in più, "
                f"oltre al modello live e allo stato dell'ottimizzatore) più forward pass extra ad "
                f"ogni generation event. Vedi docs/rt_grpo_algo.md.",
                stacklevel=2,
            )

    # ------------------------------------------------------------------
    # Hook 1: mix della history, dopo lo split nativo di TRL
    # ------------------------------------------------------------------

    def _prepare_inputs(self, generation_batch):
        if self.window_length <= 1 or not self.model.training:
            return super()._prepare_inputs(generation_batch)

        prev_buffered = self._buffered_inputs
        step_in_event = self._step % self.args.steps_per_generation
        inputs = super()._prepare_inputs(generation_batch)

        if self._buffered_inputs is not prev_buffered:
            # Nuovo generation event appena prodotto (self._buffered_inputs è stato rimpiazzato
            # dalla logica nativa in questa stessa chiamata) — salvalo per il riuso futuro.
            self._ensure_old_logps(self._buffered_inputs)
            policy_snapshot = None
            if self.is_weight_type == "bh":
                policy_snapshot = self._update_all_log_probs_bh(self._buffered_inputs)
            # Push PRIMA di mix, per contratto con MultiGenerationBuffer: da qui al prossimo push
            # history[0] è questo stesso evento, e mix() lo salta mescolando solo history[1:].
            self._gen_buffer.push_new_generation(self._buffered_inputs, policy_snapshot=policy_snapshot)

        return self._gen_buffer.mix(inputs, step_in_event)

    def _ensure_old_logps(self, slices: list[dict]) -> None:
        """
        old_per_token_logps è assente quando generazione e gradient step sono allineati (il ratio
        sarebbe 1.0 comunque per i dati correnti, vedi commento nativo in _generate_and_score_completions).
        Lo calcoliamo esplicitamente così le entry storiche hanno sempre un old_per_token_logps
        valido da riusare nel ratio di importance sampling nelle iterazioni future.
        """
        if "old_per_token_logps" in slices[0]:
            return
        with torch.no_grad():
            for s in slices:
                s["old_per_token_logps"] = self._eval_logps(s)

    def _eval_logps(self, s: dict, model=None) -> torch.Tensor:
        """Log-prob per-token di `s` sotto `model` (default self.model, altrimenti una policy congelata)."""
        model = model if model is not None else self.model
        device = next(model.parameters()).device
        input_ids = torch.cat([s["prompt_ids"], s["completion_ids"]], dim=1).to(device)
        attention_mask = torch.cat([s["prompt_mask"], s["completion_mask"]], dim=1).to(device)
        logits_to_keep = s["completion_ids"].size(1)
        logps, _, _ = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            self.args.per_device_train_batch_size,
        )
        return logps

    # ------------------------------------------------------------------
    # Balance heuristic: matrice all_log_probs (rows, T, window_length)
    # ------------------------------------------------------------------

    def _update_all_log_probs_bh(self, fresh_slices: list[dict]):
        """
        Segue lo stesso schema di MultiRolloutBuffer.update_all_log_probs in rt_ppo, adattato al
        caso per-token (matrice (rows, T, window_length) invece di (rows, window_length)) e a
        policy storiche che sono copie complete del modello, non piccole policy MLP:

        1. Congela una copia della policy che ha appena generato fresh_slices (self.model in
           questo esatto momento — nessun training step è ancora avvenuto su questi dati).
        2. Riempie la matrice di fresh_slices: colonna 0 = la sua stessa old_per_token_logps (già
           disponibile, nessun forward pass extra), colonne successive = valutazione di ciascuna
           policy storica ancora in finestra su questi dati nuovi (un forward pass per policy).
        3. Retroattivamente, per ogni evento storico ancora in finestra, valuta la policy appena
           congelata sui suoi dati e prepende una nuova colonna alla sua matrice, troncando a
           window_length (la colonna più vecchia esce dalla finestra).

        Questo metodo gira PRIMA del push del nuovo evento: la history contiene quindi solo eventi
        precedenti, fino a window_length. L'eventuale window_length-esimo sta per essere espulso
        dal push imminente e non condividerà mai la finestra con fresh_slices: entrambi i loop si
        fermano ai primi window_length-1 eventi (la sua policy è comunque già stata azzerata dal
        push precedente, vedi MultiGenerationBuffer.push_new_generation).

        Ritorna la policy congelata, da passare a MultiGenerationBuffer.push_new_generation così
        resta disponibile per valutare i prossimi window_length-1 generation event.
        """
        policy_snapshot = copy.deepcopy(self.accelerator.unwrap_model(self.model)).eval().to("cpu")
        for p in policy_snapshot.parameters():
            p.requires_grad_(False)

        kept_history = list(itertools.islice(self._gen_buffer.history, self.window_length - 1))

        with torch.no_grad():
            for s in fresh_slices:
                rows, T = s["completion_ids"].shape
                all_lp = torch.full((rows, T, self.window_length), float("-inf"), dtype=torch.float32)
                all_lp[:, :, 0] = s["old_per_token_logps"]
                for i, snapshot in enumerate(kept_history):
                    if snapshot.policy_snapshot is None:
                        continue
                    all_lp[:, :, i + 1] = self._eval_logps(s, model=snapshot.policy_snapshot)
                s["all_log_probs"] = all_lp

            for snapshot in kept_history:
                for s in snapshot.slices:
                    new_col = self._eval_logps(s, model=policy_snapshot).unsqueeze(-1)
                    s["all_log_probs"] = torch.cat([new_col, s["all_log_probs"]], dim=2)[:, :, : self.window_length]

        return policy_snapshot

    @staticmethod
    def _bh_effective_old_logps(all_log_probs: torch.Tensor) -> torch.Tensor:
        """
        Media in log-space delle log-prob sotto tutte le policy valide della finestra (-inf = non
        valutata). Un token realmente generato ha SEMPRE almeno la propria colonna valida (colonna
        0); "nessuna colonna valida" può capitare solo sul padding T introdotto da
        MultiGenerationBuffer.mix() — posizioni sempre escluse a valle da completion_mask. Lì il
        risultato naturale sarebbe -inf: sostituito con 0.0 (finito, ininfluente perché mascherato)
        per evitare che 0 * inf produca NaN quando completion_mask le azzera nella loss.
        """
        valid = torch.isfinite(all_log_probs)
        counts = valid.sum(dim=-1)
        log_mean = torch.logsumexp(all_log_probs, dim=-1) - torch.log(counts.clamp(min=1).float())
        return torch.where(counts > 0, log_mean, torch.zeros_like(log_mean))

    @staticmethod
    def _naive_old_logps(per_token_logps: torch.Tensor, inputs: dict) -> torch.Tensor:
        old = inputs.get("old_per_token_logps")
        return per_token_logps.detach() if old is None else old

    def _reduce_log_ratio(self, log_ratio: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Riduce log_ratio (per-token) secondo importance_sampling_level, come in GRPOTrainer._compute_loss."""
        if self.importance_sampling_level == "token":
            return log_ratio
        elif self.importance_sampling_level == "sequence":
            seq = (log_ratio * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
            return seq.unsqueeze(-1)
        raise ValueError(
            f"Unknown importance sampling level: {self.importance_sampling_level}. Possible values are 'token' "
            "and 'sequence'."
        )

    def _compute_ratio(self, per_token_logps: torch.Tensor, mask: torch.Tensor, inputs: dict):
        """
        Ratio di importance sampling operativo per RT-GRPO — analogo a RT_PPO._compute_ratio in
        rt_ppo.py: un unico punto, con un if interno su is_weight_type, da estendere per
        aggiungere altri tipi di correzione IS senza toccare _compute_loss_with_diagnostics.

        Ritorna (old_per_token_logps, log_ratio, coef_1): old_per_token_logps serve anche al
        fallback nativo di off_policy_mask_threshold, log_ratio è per-token grezzo (serve anche a
        vespo), coef_1 è già ridotto secondo importance_sampling_level.
        """
        if self.is_weight_type == "naive":
            old_per_token_logps = self._naive_old_logps(per_token_logps, inputs)
        elif self.is_weight_type == "bh":
            if "all_log_probs" in inputs:
                old_per_token_logps = self._bh_effective_old_logps(inputs["all_log_probs"])
            else:
                # Nessuna history disponibile per questo batch (es. primissimo step): bh degenera a naive.
                old_per_token_logps = self._naive_old_logps(per_token_logps, inputs)
        else:
            raise ValueError(f"is_weight_type sconosciuto: {self.is_weight_type!r}")

        log_ratio = per_token_logps - old_per_token_logps
        coef_1 = torch.exp(self._reduce_log_ratio(log_ratio, mask))
        return old_per_token_logps, log_ratio, coef_1

    # ------------------------------------------------------------------
    # Hook 2: loss nativa + diagnostica per finestra (nessuna reimplementazione del clipping)
    # ------------------------------------------------------------------

    def _compute_loss(self, model, inputs):
        if self.window_length <= 1 or "window_id" not in inputs:
            return super()._compute_loss(model, inputs)
        return self._compute_loss_with_diagnostics(model, inputs)

    def _compute_loss_with_diagnostics(self, model, inputs):
        """
        Copia adattata di GRPOTrainer._compute_loss (trl==1.7.0, grpo_trainer.py:2627-2832).
        La logica di loss/clipping è INVARIATA: l'unica aggiunta è la chiamata a
        _log_window_diagnostics prima del return, per riusare i tensori intermedi (coef_1, mask,
        advantages) senza un secondo forward pass. Se TRL aggiorna _compute_loss, questo metodo va
        ri-sincronizzato (vedi warning in __init__).
        """
        window_id = inputs["window_id"]

        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        mask = completion_mask if "tool_mask" not in inputs else completion_mask * inputs["tool_mask"]

        per_token_logps, entropies, aux_loss = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=True,
            compute_aux_loss=self.aux_loss_enabled,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            num_images=inputs.get("num_images"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            spatial_shapes=inputs.get("spatial_shapes"),
            num_tiles=inputs.get("num_tiles"),
            image_sizes=inputs.get("image_sizes"),
            token_type_ids=inputs.get("token_type_ids"),
            mm_token_type_ids=inputs.get("mm_token_type_ids"),
            image_position_ids=inputs.get("image_position_ids"),
        )

        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(entropies, mask, 1 - self.top_entropy_quantile)
        else:
            entropy_mask = None

        advantages = inputs["advantages"]
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(1)

        # --- Aggiunta RT-GRPO: ratio operativo (naive o bh, vedi _compute_ratio) ---
        old_per_token_logps, log_ratio, coef_1 = self._compute_ratio(per_token_logps, mask, inputs)

        if self.off_policy_mask_threshold is not None:
            sampling_per_token_logps = inputs.get("sampling_per_token_logps", old_per_token_logps)
            off_policy_mask = self.get_off_policy_mask(
                advantages=advantages,
                per_token_logps=per_token_logps,
                sampling_per_token_logps=sampling_per_token_logps,
                mask=mask,
                off_policy_threshold=self.off_policy_mask_threshold,
            )

        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )
            if self.args.use_bias_correction_kl:
                per_token_kl = per_token_kl * coef_1

        if self.loss_type == "cispo":
            clamped_ratios = torch.clamp(coef_1, max=self.epsilon_high).detach()
            per_token_loss = -clamped_ratios * advantages * per_token_logps
        elif self.loss_type in ["grpo", "bnpo", "dr_grpo", "dapo", "luspo"]:
            coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
            if self.args.delta is not None:
                coef_1 = torch.clamp(coef_1, max=self.args.delta)
            per_token_loss1 = coef_1 * advantages
            per_token_loss2 = coef_2 * advantages
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        elif self.loss_type == "sapo":
            temperatures = torch.where(advantages > 0, self.args.sapo_temperature_pos, self.args.sapo_temperature_neg)
            soft_coef_1 = torch.sigmoid(temperatures * (coef_1 - 1)) * 4 / temperatures
            per_token_loss = -soft_coef_1 * advantages
        elif self.loss_type == "vespo":
            phi_seq = self.get_gamma_weights(
                advantages=advantages,
                log_ratio_per_token=log_ratio,
                mask=mask,
                importance_sampling_ratio=inputs.get("importance_sampling_ratio"),
                k_pos=self.args.vespo_k_pos,
                lambda_pos=self.args.vespo_lambda_pos,
                k_neg=self.args.vespo_k_neg,
                lambda_neg=self.args.vespo_lambda_neg,
            )
            per_token_loss = -phi_seq * advantages * per_token_logps
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        if self.off_policy_mask_threshold is not None:
            per_token_loss = per_token_loss * off_policy_mask

        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask

        if self.use_vllm and self.vllm_importance_sampling_correction and self.loss_type != "vespo":
            per_token_loss = per_token_loss * inputs["importance_sampling_ratio"]

        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        mode = "train" if self.model.training else "eval"
        if self.loss_type in ["grpo", "sapo"]:
            loss = ((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
            normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0
            loss = loss / normalizer
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * mask).sum() / mask.sum().clamp(min=1.0)
            normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0
            loss = loss / normalizer
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
            normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0
            loss = loss / normalizer
        elif self.loss_type in ["cispo", "dapo", "vespo"]:
            normalizer = inputs["num_items_in_batch"] / self.accelerator.num_processes
            loss = (per_token_loss * mask).sum() / normalizer
        elif self.loss_type == "luspo":
            loss = (per_token_loss * mask.sum(1, keepdim=True)).mean()
            normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0
            loss = loss / normalizer
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        if self.aux_loss_enabled:
            normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0
            loss = loss + self.router_aux_loss_coef * aux_loss / normalizer
            self._metrics[mode]["aux_loss"].append(self.accelerator.gather_for_metrics(aux_loss).mean().item())

        completion_token_count = mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:
                return x.mean()
            else:
                return (x * mask).sum() / completion_token_count

        if self.beta != 0.0:
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        mean_entropy = masked_batch_mean(entropies)
        self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())

        if self.loss_type in ["grpo", "bnpo", "dr_grpo", "dapo", "luspo"]:
            is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages < 0)
            is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages > 0)
            is_region_clipped = is_low_clipped | is_high_clipped

            low_clip = masked_batch_mean(is_low_clipped.float())
            high_clip = masked_batch_mean(is_high_clipped.float())
            clip_ratio = masked_batch_mean(is_region_clipped.float())

            gathered_low_clip = self.accelerator.gather(low_clip)
            self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
            self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
            gathered_high_clip = self.accelerator.gather(high_clip)
            self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
            self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
            gathered_clip_ratio = self.accelerator.gather(clip_ratio)
            self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())
        elif self.loss_type == "cispo":
            is_cispo_clipped = (coef_1 > self.epsilon_high) & (advantages > 0)
            cispo_clip_ratio = masked_batch_mean(is_cispo_clipped.float())
            gathered_cispo_clip_ratio = self.accelerator.gather(cispo_clip_ratio)
            self._metrics[mode]["cispo_clip_ratio"].append(gathered_cispo_clip_ratio.nanmean().item())
        elif self.loss_type == "vespo":
            gathered_phi_seq = self.accelerator.gather(phi_seq)
            self._metrics[mode]["vespo/phi_seq_mean"].append(gathered_phi_seq.nanmean().item())

        # --- Aggiunta RT-GRPO: diagnostica per finestra/età dei dati (docs/rt_grpo_algo.md) ---
        # Se bh è attivo, coef_1 è già quello operativo (bh); calcoliamo qui anche il ratio naive
        # (mai usato per la loss in quel caso) solo per il confronto in diagnostica, riusando gli
        # stessi helper di _compute_ratio.
        naive_coef_1 = None
        if self.is_weight_type == "bh" and "all_log_probs" in inputs:
            naive_log_ratio = per_token_logps - self._naive_old_logps(per_token_logps, inputs)
            naive_coef_1 = torch.exp(self._reduce_log_ratio(naive_log_ratio, mask))

        self._log_window_diagnostics(coef_1, mask, window_id, mode, naive_coef_1=naive_coef_1)

        return loss

    def _log_window_diagnostics(
        self,
        coef_1: torch.Tensor,
        mask: torch.Tensor,
        window_id: torch.Tensor,
        mode: str,
        naive_coef_1: torch.Tensor = None,
    ) -> None:
        """
        Logga, per ciascun window_id (0=corrente, 1..W-1=storico), le metriche già usate da
        rt_ppo per capire quanto i dati riusati restano utili al gradiente: |ratio-1|, KL
        approssimata (Schulman), varianza del ratio, Effective Sample Size, clip fraction
        (quest'ultima solo per i loss_type basati su clipping PPO-style), numero di campioni.

        Se `naive_coef_1` è None, `coef_1` è il ratio naive (bh non attivo) e viene loggato con i
        nomi base (diagnostics_*/window_N). Se `naive_coef_1` è fornito (bh attivo), `coef_1` è il
        ratio bh operativo: logghiamo il naive con i nomi base (com'era la policy comportamentale
        "grezza" per riga) e il bh con suffisso `_bh`, per confrontarli nello stesso run.

        window_id/coef_1 hanno granularità per riga (una per completion), quindi il ratio è
        ridotto a un valore per riga (media mascherata sui token) prima di raggruppare per finestra.
        """
        with torch.no_grad():
            def reduce_to_row(c: torch.Tensor) -> torch.Tensor:
                return c.squeeze(-1) if c.shape[1] == 1 else (c * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)

            ratio_variants = [("", reduce_to_row(naive_coef_1 if naive_coef_1 is not None else coef_1))]
            if naive_coef_1 is not None:
                ratio_variants.append(("_bh", reduce_to_row(coef_1)))

            do_clip_diag = self.loss_type in ("grpo", "bnpo", "dr_grpo", "dapo", "luspo")

            for tag, row_ratio in ratio_variants:
                for wid in window_id.unique().tolist():
                    wid = int(wid)
                    w_ratio = row_ratio[window_id == wid]
                    if w_ratio.numel() == 0:
                        continue
                    suffix = f"window_{wid}{tag}"
                    self._metrics[mode][f"diagnostics_abs_ratio/{suffix}"].append((w_ratio - 1).abs().mean().item())
                    self._metrics[mode][f"diagnostics_kl/{suffix}"].append(
                        ((w_ratio - 1) - torch.log(w_ratio)).mean().item()
                    )
                    self._metrics[mode][f"diagnostics_var/ratio_var_{suffix}"].append(
                        w_ratio.var().item() if w_ratio.numel() > 1 else 0.0
                    )
                    ess = (w_ratio.sum() ** 2 / (w_ratio ** 2).sum()) / w_ratio.numel()
                    self._metrics[mode][f"diagnostics_ess/ess_{suffix}"].append(ess.item())
                    self._metrics[mode][f"diagnostics_age/n_samples_{suffix}"].append(float(w_ratio.numel()))
                    if do_clip_diag:
                        clipped = (w_ratio < 1 - self.epsilon_low) | (w_ratio > 1 + self.epsilon_high)
                        self._metrics[mode][f"diagnostics_clip/clip_fraction_{suffix}"].append(clipped.float().mean().item())
