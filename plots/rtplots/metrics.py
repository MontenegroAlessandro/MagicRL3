"""Catalogo delle metriche plottabili.

Due fonti:
  - `local`: i file evaluations.npz scritti da EvalCallback (veloci, sempre completi);
  - `wandb`: la history del run (tutto il resto), scaricata una volta e messa in cache.

Ogni voce: chiave logica -> etichetta nella UI, etichetta dell'asse y, fonte e,
per le metriche locali, il campo del .npz da cui si legge.
"""
from __future__ import annotations

# (gruppo, [(chiave, etichetta UI, etichetta asse y, fonte, campo npz)])
METRIC_GROUPS = [
    ("Eval", [
        ("eval/mean_reward", "Mean return", "Mean Return", "local", "results"),
        ("eval/mean_ep_length", "Lunghezza episodio", "Episode Length", "local", "ep_lengths"),
    ]),
    ("Rollout", [
        ("rollout/ep_rew_mean", "Return di training", "Training Return", "wandb", None),
        ("rollout/ep_len_mean", "Lunghezza episodio (training)", "Episode Length", "wandb", None),
        ("time/fps", "FPS", "FPS", "wandb", None),
    ]),
    ("Ottimizzazione", [
        ("train/loss", "Loss totale", "Loss", "wandb", None),
        ("train/policy_gradient_loss", "Loss policy", "Policy Loss", "wandb", None),
        ("train/value_loss", "Loss value", "Value Loss", "wandb", None),
        ("train/entropy_loss", "Loss entropia", "Entropy Loss", "wandb", None),
        ("train/explained_variance", "Explained variance", "Explained Variance", "wandb", None),
        ("train/clip_fraction", "Clip fraction", "Clip Fraction", "wandb", None),
        ("train/learning_rate", "Learning rate", "Learning Rate", "wandb", None),
        ("train/std", "Std della policy", "Policy Std", "wandb", None),
        ("train/approx_kl_window_all_mean", "KL approx (finestra)", "Approx KL", "wandb", None),
    ]),
    ("Diagnostiche IS", [
        ("diagnostics_kl/kl_mean", "KL medio (fine update)", "KL", "wandb", None),
        ("diagnostics_kl/initial_kl_mean", "KL medio (inizio update)", "KL", "wandb", None),
        ("diagnostics_ess/final_naive_ess_mean", "ESS (fine update)", "ESS", "wandb", None),
        ("diagnostics_ess/initial_naive_ess_mean", "ESS (inizio update)", "ESS", "wandb", None),
        ("diagnostics_clip/clip_fraction_mean", "Clip fraction (finestra)",
         "Clip Fraction", "wandb", None),
        ("diagnostics_var/final_naive_ratio_var_mean", "Varianza dei ratio",
         "Ratio Variance", "wandb", None),
        ("diagnostics_abs_ratio/final_mean", "|ratio| medio", "Mean |ratio|", "wandb", None),
        ("debug/rollout_trajectories", "Traiettorie per rollout", "Trajectories", "wandb", None),
    ]),
]

METRICS = {
    key: {"key": key, "label": label, "ylabel": ylabel, "source": source, "npz_field": field}
    for _, items in METRIC_GROUPS for key, label, ylabel, source, field in items
}

# chiave -> campo del .npz, per le metriche leggibili in locale
LOCAL_FIELDS = {k: v["npz_field"] for k, v in METRICS.items() if v["source"] == "local"}

DEFAULT_METRIC = "eval/mean_reward"


def metric_info(key: str) -> dict:
    """Voce del catalogo; per una chiave W&B non elencata restituisce un default sensato."""
    if key in METRICS:
        return METRICS[key]
    return {"key": key, "label": key, "ylabel": key.split("/")[-1].replace("_", " "),
            "source": "wandb", "npz_field": None}


def ui_groups() -> list[dict]:
    """Struttura per le tendine della pagina."""
    return [{"group": group,
             "options": [{"key": k, "label": lab, "source": src} for k, lab, _, src, _ in items]}
            for group, items in METRIC_GROUPS]
