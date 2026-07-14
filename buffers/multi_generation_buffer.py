import itertools
from collections import deque
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


# Chiavi allineate al prompt: shape (B, P), left-padded da TRL.
_PROMPT_KEYS = {"prompt_ids", "prompt_mask"}

# Chiavi allineate alla completion: shape (B, C), right-padded da TRL.
_COMPLETION_KEYS = {
    "completion_ids",
    "completion_mask",
    "old_per_token_logps",
    "ref_per_token_logps",
    "sampling_per_token_logps",
    "tool_mask",
}

# Matrice per la correzione bh: shape (B, T, window_length), log-prob di ogni token sotto ogni
# policy della finestra (-inf dove non ancora valutata). T va paddato come le chiavi completion,
# ma l'ultima dimensione (window) NON va paddata: è sempre allocata a window_length di default
# (vedi RT_GRPOTrainer._update_all_log_probs_bh), quindi F.pad va sulla dimensione T (la seconda),
# non sull'ultima come per le chiavi 2D sopra.
_LOGPROB_MATRIX_KEYS = {"all_log_probs"}

# Chiavi scalari/metadati: non vengono paddate né concatenate riga per riga, sono
# ricalcolate esplicitamente in mix() (num_items_in_batch) o ignorate.
_SCALAR_KEYS = {"num_items_in_batch"}


@dataclass
class GenerationSnapshot:
    """
    Snapshot su CPU di un intero generation event, già tagliato nelle sue steps_per_generation fette.

    policy_snapshot: deepcopy congelata del modello che ha generato questo evento (solo se
    is_weight_type="bh"), riusata per valutare retroattivamente eventi futuri finché resta in
    history. Viene azzerata esplicitamente da push_new_generation quando l'entry diventa la più
    vecchia di una deque piena (la sua policy non serve più a nessuna valutazione bh futura).
    """
    slices: list[dict[str, torch.Tensor]]
    policy_snapshot: Any = None


class MultiGenerationBuffer:
    """
    Tiene in memoria gli ultimi window_length generation event di GRPO, incluso quello corrente
    (vedi docs/rt_grpo_algo.md). Contratto con RT_GRPOTrainer._prepare_inputs: push_new_generation
    viene chiamato al detection di ogni nuovo evento, PRIMA di qualsiasi mix() — quindi history[0]
    è sempre l'evento corrente, e mix() lo salta mescolando la fetta corrente solo con history[1:]
    (i window_length-1 eventi precedenti).

    Non è un buffer stile SB3 (get()/add() a generatore): TRL non ha un punto di iniezione
    equivalente a rollout_buffer_class, quindi l'oggetto è usato internamente da RT_GRPOTrainer
    con un'API minimale (push_new_generation/mix), chiamata una volta per step da _prepare_inputs.
    """

    def __init__(self, window_length: int, pad_token_id: int):
        self.window_length = window_length
        self.pad_token_id = pad_token_id
        # maxlen = window_length: uno slot per l'evento corrente (history[0]) + window_length-1 storici.
        self.history: deque[GenerationSnapshot] = deque(maxlen=window_length)

    def push_new_generation(self, slices: list[dict[str, torch.Tensor]], policy_snapshot: Any = None) -> None:
        """Salva su CPU l'intero generation event appena prodotto (le sue steps_per_generation fette)."""
        snapshot = GenerationSnapshot(
            slices=[
                {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v) for k, v in s.items()}
                for s in slices
            ],
            policy_snapshot=policy_snapshot,
        )
        self.history.appendleft(snapshot)
        # A deque piena, l'entry più vecchia rimasta (indice -1) non condividerà mai più la finestra
        # con un evento fresco: la sua policy congelata non serve più a nessuna valutazione bh (vedi
        # i loop limitati a window_length-1 in _update_all_log_probs_bh). Liberarla subito tiene in
        # RAM al massimo window_length-1 copie del modello, come dichiarato dal warning in __init__.
        # I suoi dati (slices) restano: vengono ancora mescolati da mix() finché l'entry è in history.
        if len(self.history) == self.history.maxlen:
            self.history[-1].policy_snapshot = None

    def mix(self, current_slice: dict[str, torch.Tensor], step_in_event: int) -> dict[str, torch.Tensor]:
        """
        Mescola la fetta corrente (già pronta per questo step) con la fetta allo stesso step_in_event
        di ciascun generation event storico. Aggiunge on_policy_mask/window_id e ricalcola
        num_items_in_batch sui token di completion effettivamente presenti nel batch misto.

        Nota: gli snapshot storici hanno sempre esattamente steps_per_generation fette (sono stati
        salvati dopo lo stesso split nativo), quindi step_in_event indicizza in modo consistente
        sia la fetta corrente che quelle storiche.
        """
        device = next(v.device for v in current_slice.values() if isinstance(v, torch.Tensor))

        # Per contratto history[0] è l'evento CORRENTE (pushato al detection, prima di ogni mix):
        # va saltato, altrimenti la fetta corrente comparirebbe due volte nel batch misto.
        # history[1] = generation event immediatamente precedente, ecc.
        historical_slices = [
            {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in snapshot.slices[step_in_event].items()}
            for snapshot in itertools.islice(self.history, 1, None)
        ]

        all_slices = [current_slice] + historical_slices
        merged = self._pad_and_concat(all_slices)

        row_counts = [self._n_rows(s) for s in all_slices]
        window_id = torch.cat([
            torch.full((n,), i, dtype=torch.long, device=device) for i, n in enumerate(row_counts)
        ])
        merged["window_id"] = window_id
        merged["on_policy_mask"] = (window_id == 0).float()

        # Ricalcolato sui token di completion realmente presenti in QUESTO batch misto (somma di
        # completion_mask), non sommando i num_items_in_batch "per intero evento" delle fette
        # storiche: quel valore rappresenta il totale dell'intero generation event storico, non
        # della sola fetta step_in_event inclusa qui, e sommarlo sovrastimerebbe il normalizzatore.
        # Vedi docs/rt_grpo_algo.md.
        merged["num_items_in_batch"] = merged["completion_mask"].sum()

        return merged

    @staticmethod
    def _n_rows(slice_dict: dict) -> int:
        return next(v.size(0) for v in slice_dict.values() if isinstance(v, torch.Tensor))

    def _pad_and_concat(self, slices: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """
        Concatena `slices` lungo dim 0. I campi allineati al prompt sono left-pad, quelli allineati
        alla completion sono right-pad, fino alla lunghezza massima tra tutte le fette (convenzione
        TRL). I campi scalari/non-tensoriali sono presi dalla fetta corrente (la prima).
        """
        prompt_len = max(s["prompt_ids"].size(1) for s in slices)
        comp_len = max(s["completion_ids"].size(1) for s in slices)

        result: dict[str, torch.Tensor] = {}
        current = slices[0]
        for key in current:
            if key in _SCALAR_KEYS:
                continue
            tensors = []
            for s in slices:
                t = s.get(key)
                if not isinstance(t, torch.Tensor):
                    continue

                if key in _PROMPT_KEYS:
                    pad = prompt_len - t.size(1)
                    if pad > 0:
                        fill = self.pad_token_id if key == "prompt_ids" else 0
                        t = F.pad(t, (pad, 0), value=fill)
                elif key in _COMPLETION_KEYS:
                    pad = comp_len - t.size(1)
                    if pad > 0:
                        fill = self.pad_token_id if key == "completion_ids" else 0.0
                        t = F.pad(t, (0, pad), value=fill)
                elif key in _LOGPROB_MATRIX_KEYS:
                    # (B, T, window_length): pad va sulla dimensione T (la seconda-ultima), non
                    # sull'ultima (window) — F.pad specifica le coppie (sinistra,destra) partendo
                    # dall'ultima dimensione, quindi (0,0) per window e (0,pad) per T.
                    pad = comp_len - t.size(1)
                    if pad > 0:
                        t = F.pad(t, (0, 0, 0, pad), value=float("-inf"))

                tensors.append(t)
            if tensors:
                result[key] = torch.cat(tensors, dim=0)

        # Metadati non concatenabili riga per riga: presi dalla fetta corrente.
        for key, val in current.items():
            if key not in result and key not in _SCALAR_KEYS:
                result[key] = val

        return result
