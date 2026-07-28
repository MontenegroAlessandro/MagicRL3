"""Valori di base per environment, letti dai config veri.

`n_steps`, `batch_size` e `n_epochs` di `run/config/ppo/ppo_<env>.yaml` servono a
due cose nell'indice: dedurre il `setting` quando mancano i tag e calcolare
`epoch_mult` (le epoche in multipli del valore base, cio' che distingue le
baseline PPO fra loro).

Prima erano ricopiati a mano in index.py: due copie della stessa verita', con il
rischio che un config cambi e l'intero indice resti classificato con i valori
vecchi senza che nulla lo segnali. Qui si leggono dai file, indicizzati per
`experiment.env_name` (ogni yaml e' autosufficiente su queste chiavi).
"""
from __future__ import annotations

from functools import lru_cache

from .paths import PPO_CONFIG_DIR

KEYS = ("n_steps", "batch_size", "n_epochs")


@lru_cache(maxsize=1)
def env_base() -> dict[str, dict]:
    """{env_name: {n_steps, batch_size, n_epochs}} dai config PPO."""
    import yaml

    out: dict[str, dict] = {}
    for path in sorted(PPO_CONFIG_DIR.glob("ppo_*.yaml")):
        try:
            exp = (yaml.safe_load(path.read_text()) or {}).get("experiment") or {}
        except yaml.YAMLError:
            continue
        env = exp.get("env_name")
        if not env:
            continue
        values = {k: exp.get(k) for k in KEYS}
        if any(v is None for v in values.values()):
            continue
        # ppo_default.yaml ha lo stesso env_name di ppo_halfcheetah.yaml: vince
        # il file specifico dell'environment, non il default.
        if env in out and path.stem == "ppo_default":
            continue
        out[env] = values
    return out


def base_for(env: str | None) -> dict:
    return env_base().get(env) or {}
