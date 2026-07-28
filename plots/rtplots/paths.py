"""Percorsi condivisi da tutti gli script di plotting.

Regola: gli artefatti pesanti (cache dell'indice W&B, curve scaricate) vivono in
/storage, mai nella repo. Nella repo finiscono solo gli script e, di default, le
figure generate (plots/output/, in .gitignore).

Quali progetti W&B esistono e come si leggono non sta qui ma in `sources/`.
"""
from __future__ import annotations

import os
from pathlib import Path

# Radice della repo (plots/rtplots/paths.py -> plots/ -> repo/)
REPO_ROOT = Path(__file__).resolve().parents[2]
PLOTS_ROOT = REPO_ROOT / "plots"
# Config veri degli esperimenti: da qui si leggono i valori base per environment
PPO_CONFIG_DIR = REPO_ROOT / "run" / "config" / "ppo"

# Cache: sovrascrivibile con RTPLOTS_CACHE
CACHE_DIR = Path(os.environ.get("RTPLOTS_CACHE", "/storage/fis1/plots_cache"))
INDEX_PARQUET = CACHE_DIR / "run_index.parquet"
INDEX_CSV = CACHE_DIR / "run_index.csv"
EVAL_MAP_JSON = CACHE_DIR / "eval_paths.json"
CURVE_DIR = CACHE_DIR / "curves"
# Selezione salvata dal selettore interattivo (plots/scripts/selector.py):
# selection.json e' sempre l'ultima salvata, selections/ tiene lo storico per nome.
SELECTION_JSON = CACHE_DIR / "selection.json"
SELECTIONS_DIR = CACHE_DIR / "selections"

# Output figure: sovrascrivibile con RTPLOTS_OUTPUT o --outdir
OUTPUT_DIR = Path(os.environ.get("RTPLOTS_OUTPUT", PLOTS_ROOT / "output"))

# Dove cercare le cartelle di campagna (contengono logs/<run_id>/evaluations.npz)
STORAGE_ROOTS = [Path(p) for p in os.environ.get(
    "RTPLOTS_STORAGE", "/storage/fis1"
).split(":")]

WANDB_ENTITY = os.environ.get("RTPLOTS_WANDB_ENTITY", "alessandro-montenegro-polimi")


def wandb_path(project: str) -> str:
    """entity/project per un progetto dato (i run_id sono unici per progetto)."""
    return project if "/" in project else f"{WANDB_ENTITY}/{project}"


def ensure_dirs() -> None:
    for d in (CACHE_DIR, CURVE_DIR, SELECTIONS_DIR, OUTPUT_DIR):
        d.mkdir(parents=True, exist_ok=True)
