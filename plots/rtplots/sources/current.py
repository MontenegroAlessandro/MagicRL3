"""Campagne attuali: config strutturata in `experiment`, convenzioni correnti.

E' la fonte di tutto quello che il codice logga oggi: le colonne dell'indice
arrivano dalla config senza traduzioni. Un progetto W&B vale l'altro, quindi
aggiungerne uno e' una riga in `PROJECTS` — la fonte descrive *le convenzioni*,
non una campagna.

Convenzioni date per buone qui dentro:
  - `experiment` contiene tutti i parametri (`api.runs()` la restituisce vuota:
    serve `run.load(force=True)`, che `index.py` fa sempre);
  - il campionamento dei minibatch e' `batch_sampling` (random|balanced|weighted);
  - i tag sono quattro: SHA del codice, setting, famiglia, campagna numerata —
    di questi contano `setting<N>` e i tag di ablation qui sotto, il resto e'
    gia' nella config;
  - `dir_name` finisce in /storage e il suo basename e' il nome della campagna.
"""
from __future__ import annotations

import os

from .base import RunSource

# Tag che marcano run fuori dai sottospazi di STATUS_EXP (ablation sul clip
# range, configurazioni tunate): filtrare `ablation=none` isola i sottospazi.
ABLATION_TAGS = ("clip_range_ablation2", "clip_range_ablation", "tuned_1", "tuned_2")

# progetto W&B -> environment atteso ("" = lo dice la config, come deve essere).
# RTPLOTS_WANDB_PROJECT sovrascrive il primo, che resta il default quando una
# run non dice da quale progetto viene.
PROJECTS = {
    os.environ.get("RTPLOTS_WANDB_PROJECT", "rebuttal"): "",
    "rt-ppo-ablations": "",
}

SOURCE = RunSource(
    name="wandb",
    projects=PROJECTS,
    ablation_tags=ABLATION_TAGS,
    # i nomi delle run non codificano tutti i parametri (le campagne random e
    # balanced hanno run omonime e diverse): l'omonimia qui non vuol dire nulla
    name_is_key=False,
)
