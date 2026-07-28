"""Run del paper: un progetto W&B per environment, convenzioni precedenti.

Tutto cio' che separa queste run da quelle attuali sta qui:

  - **`api.runs()` restituisce `run.config` vuota.** I parametri arrivano solo
    dopo `run.load(force=True)`, che `index.py` chiama sempre. Chi sonda questi
    progetti senza `load()` conclude a torto che siano senza metadati e che
    vadano ricostruiti dal nome del run: non serve.
  - **I tag di setting hanno una numerazione diversa** da quella dei setting:
    `wppo_3` e' il setting **2**. Verificato su tutte le run che hanno il tag:
    la deduzione dai soli n_steps/batch_size da' sempre lo stesso valore.
  - **`adaptive_lr` non esiste nella config** perche' queste run precedono la
    feature: il valore vero e' False, non "ignoto".
  - **Le diagnostiche hanno chiavi diverse** e non tutte hanno una controparte
    con la stessa semantica: vedi UNCONFIRMED_ALIASES.

L'environment non compare nella config di alcune run vecchie: `projects` fa da
riserva, un progetto per environment.
"""
from __future__ import annotations

from ..metrics import METRICS
from .base import RunSource
from .current import ABLATION_TAGS

PROJECTS = {
    "forzaroma-rt-ppo-ant": "Ant-v5",
    "erghosting-rt-ppo-half-cheetah": "HalfCheetah-v5",
    "forzaroma-rt-ppo-hopper": "Hopper-v5",
    "forzaroma-rt-ppo-reacher": "Reacher-v5",
    "forzaroma-rt-ppo-swimmer": "Swimmer-v5",
    "forzaroma-rt-ppo-walker": "Walker2d-v5",
}

# La numerazione dei tag NON e' quella dei setting.
SETTING_TAGS = {"wppo_1": 1, "wppo_3": 2, "wppo_3_v2": 3, "wppo_3_v2_3M": 3}

# Corrispondenze plausibili ma NON confermate fra le diagnostiche di queste run
# e quelle del catalogo: `post_*` sta a meta' strada fra `initial_*` e `final_*`
# del codice attuale. Finche' la semantica non e' verificata restano fuori dagli
# alias attivi: promuoverne una qui sotto e' una decisione sperimentale, non una
# svista da correggere.
UNCONFIRMED_ALIASES = {
    "diagnostics_ess/final_naive_ess_mean": "diagnostics_ess/post_naive_mean",
    "diagnostics_kl/kl_mean": "diagnostics_kl/post_naive_mean",
}

# Le metriche train/* e rollout/* coincidono: nessun alias serve.
ALIASES: dict[str, str] = {}

_DIAGNOSTIC_PREFIXES = ("diagnostics_", "debug/")
UNAVAILABLE = {
    key: ("chiave diversa nelle run del paper e corrispondenza semantica non "
          "confermata (vedi UNCONFIRMED_ALIASES in rtplots/sources/paper.py)")
    for key in METRICS
    if key.startswith(_DIAGNOSTIC_PREFIXES) and key not in ALIASES
}

SOURCE = RunSource(
    name="paper",
    projects=PROJECTS,
    setting_tags=SETTING_TAGS,
    ablation_tags=ABLATION_TAGS,
    defaults={"adaptive_lr": False},
    metric_aliases=ALIASES,
    metrics_unavailable=UNAVAILABLE,
    name_is_key=True,
)
