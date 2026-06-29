from collections import deque
from typing import Any

import torch
import torch.nn.functional as F
from trl import GRPOTrainer


# Prompt-aligned keys: shape (B, P), left-padded by TRL.
# When mixing batches with different P, we left-pad shorter ones to match.
_PROMPT_KEYS = {"prompt_ids", "prompt_mask"}

# Completion-aligned keys: shape (B, C), right-padded by TRL.
# When mixing batches with different C, we right-pad shorter ones to match.
_COMPLETION_KEYS = {
    "completion_ids",
    "completion_mask",
    "old_per_token_logps",   # behavioral log-probs — key for IS correction
    "ref_per_token_logps",   # reference model log-probs (only if beta != 0)
    "sampling_per_token_logps",
    "tool_mask",
}


class RT_GRPOTrainer(GRPOTrainer):
    """
    GRPOTrainer with a sliding window of past generation batches.

    Standard GRPO discards each batch after the gradient update.
    This trainer keeps the last window_length-1 batches and mixes
    them with the current one before training, giving more gradient
    signal without extra generation cost.

    IS correction is handled automatically: old_per_token_logps (the
    log-probs under the behavioral policy at generation time) are stored
    for every entry. The existing PPO clipping in GRPOTrainer then limits
    the damage from stale off-policy samples.

    window_length = 1 → identical to GRPOTrainer (no history kept).

    :param window_length: Total batches to use per update (current + history).
    """

    def __init__(self, *args, window_length: int = 1, **kwargs):
        self.window_length = window_length
        # Ring buffer: holds up to window_length-1 past snapshots as CPU tensors
        self._rollout_history: deque = deque(maxlen=max(0, window_length - 1))
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # Core override
    # ------------------------------------------------------------------

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Any]]
    ) -> dict[str, Any]:
        # Fresh generation from the base trainer (sampling + scoring).
        batch = super()._generate_and_score_completions(inputs)

        if self.window_length == 1:
            return batch

        # old_per_token_logps is the behavioral log-prob π_k(a|s) evaluated
        # right after generation. The parent sets it to None when generation
        # and gradient steps are aligned (the ratio would be 1.0 anyway for
        # current data). We compute it explicitly so historical entries can
        # use the same IS pipeline in compute_loss.
        if batch.get("old_per_token_logps") is None:
            batch["old_per_token_logps"] = self._compute_behavioral_logps(batch)

        # Snapshot the current batch on CPU *before* mixing — this is what
        # we'll replay in future iterations.
        snapshot = {k: v.cpu() for k, v in batch.items() if isinstance(v, torch.Tensor)}

        # Mix current batch with accumulated history (if any).
        # We do this before appending so the snapshot doesn't double-count.
        if self._rollout_history:
            batch = self._concat_with_history(batch)

        self._rollout_history.appendleft(snapshot)
        return batch

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_behavioral_logps(self, batch: dict) -> torch.Tensor:
        """
        Compute per-token log-probs of the current model on the batch.
        Called only when the parent left old_per_token_logps = None (step-aligned case).
        The model hasn't updated yet at this point, so these are the true behavioral log-probs.
        """
        input_ids = torch.cat([batch["prompt_ids"], batch["completion_ids"]], dim=1)
        attention_mask = torch.cat([batch["prompt_mask"], batch["completion_mask"]], dim=1)
        logits_to_keep = batch["completion_ids"].size(1)

        with torch.no_grad():
            logps, _ = self._get_per_token_logps_and_entropies(
                self.model,
                input_ids,
                attention_mask,
                logits_to_keep,
                self.args.per_device_train_batch_size,
            )
        return logps

    def _concat_with_history(self, current: dict) -> dict:
        """
        Concatenate the current batch with all history entries along dim 0.

        Prompt tensors are left-padded and completion tensors are right-padded
        to the maximum length across current + history, matching how TRL pads
        within a single batch. Non-tensor fields are taken from current.
        """
        device = next(v for v in current.values() if isinstance(v, torch.Tensor)).device

        # Find the maximum prompt and completion lengths across all entries.
        # Needed because different generation steps may produce different padding sizes.
        prompt_len = current["prompt_ids"].size(1)
        comp_len = current["completion_ids"].size(1)
        for entry in self._rollout_history:
            prompt_len = max(prompt_len, entry["prompt_ids"].size(1))
            comp_len = max(comp_len, entry["completion_ids"].size(1))

        # Move history tensors back to the training device.
        all_entries = [current] + [
            {k: v.to(device) for k, v in e.items()} for e in self._rollout_history
        ]

        pad_id = self._tokenizer.pad_token_id
        result = {}

        for key in current:
            tensors = []
            for entry in all_entries:
                t = entry.get(key)
                if not isinstance(t, torch.Tensor):
                    continue

                if key in _PROMPT_KEYS:
                    # Left-pad: prompt padding is always on the left in TRL
                    pad = prompt_len - t.size(1)
                    if pad > 0:
                        fill = pad_id if key == "prompt_ids" else 0
                        t = F.pad(t, (pad, 0), value=fill)

                elif key in _COMPLETION_KEYS:
                    # Right-pad: completion padding is always on the right in TRL
                    pad = comp_len - t.size(1)
                    if pad > 0:
                        fill = pad_id if key == "completion_ids" else 0.0
                        t = F.pad(t, (0, pad), value=fill)

                # 1-D tensors (e.g. advantages) and matched-length tensors are cat'd as-is.
                tensors.append(t)

            if tensors:
                result[key] = torch.cat(tensors, dim=0)

        # Pass through non-tensor metadata from the current batch (e.g. prompts as strings).
        for key, val in current.items():
            if key not in result:
                result[key] = val

        return result
