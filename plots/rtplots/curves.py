"""Caricamento e aggregazione delle curve di eval.

Fonte primaria: i file locali <campagna>/logs/<run_id>/evaluations.npz scritti da
EvalCallback (timesteps, results[n_eval, n_episodi]). Sono completi e veloci.
Fallback: la history W&B, messa in cache in /storage come parquet e, dentro un
processo vivo (il selettore), anche in memoria.

Le run del paper non hanno .npz locali: le loro curve arrivano sempre da W&B.
Vivendo in progetti diversi (i run_id sono unici per progetto, non per entity),
il progetto fa parte della chiave di ogni cache e del nome della metrica: quale
chiave sia davvero loggata lo dice la fonte (`sources/`).
"""
from __future__ import annotations

import json
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import sources
from .metrics import DEFAULT_METRIC, LOCAL_FIELDS
from .paths import (CURVE_DIR, EVAL_MAP_JSON, STORAGE_ROOTS, ensure_dirs,
                    wandb_path)

# Cache in memoria condivisa (serve al selettore, che ridisegna in continuazione).
# La chiave include il progetto: i run_id sono unici solo al suo interno.
_MEM: OrderedDict[tuple, pd.DataFrame] = OrderedDict()
MAX_MEM_CURVES = 4000


def clear_cache() -> None:
    _MEM.clear()


def _mem_get(key):
    df = _MEM.get(key)
    if df is not None:
        _MEM.move_to_end(key)
    return df


def _mem_put(key, df) -> None:
    _MEM[key] = df
    while len(_MEM) > MAX_MEM_CURVES:
        _MEM.popitem(last=False)


# --- mappa run_id -> evaluations.npz ---------------------------------------

def build_eval_map(refresh: bool = False, verbose: bool = False) -> dict[str, str]:
    """Indicizza tutti gli evaluations.npz presenti in storage."""
    ensure_dirs()
    if EVAL_MAP_JSON.exists() and not refresh:
        return json.loads(EVAL_MAP_JSON.read_text())
    mapping: dict[str, str] = {}
    for root in STORAGE_ROOTS:
        if not root.exists():
            continue
        # campagne dirette (<root>/<campagna>/logs/<id>) e annidate (<root>/<c>/<sub>/logs/<id>)
        for pattern in ("*/logs/*/evaluations.npz", "*/*/logs/*/evaluations.npz"):
            for path in root.glob(pattern):
                mapping[path.parent.name] = str(path)
    EVAL_MAP_JSON.write_text(json.dumps(mapping, indent=0))
    if verbose:
        print(f"[curves] {len(mapping)} evaluations.npz indicizzati in {EVAL_MAP_JSON}")
    return mapping


# --- lettura di una singola curva ------------------------------------------

def curve_from_npz(path: str | Path, field: str = "results") -> pd.DataFrame:
    """DataFrame [step, ret, ret_std_eps] dalla media sugli episodi di eval.

    `field`: 'results' (return) oppure 'ep_lengths' (lunghezza degli episodi).
    """
    data = np.load(path)
    values = data[field]  # (n_eval, n_episodi)
    return pd.DataFrame({
        "step": data["timesteps"].astype(float),
        "ret": values.mean(axis=1),
        "ret_std_eps": values.std(axis=1),
        # quanti episodi stanno dietro a ogni punto: serve a separare il rumore
        # dello stimatore dall'instabilita' vera (vedi summary.instability)
        "n_eps": float(values.shape[1]),
    })


def curve_from_wandb(run_id: str, project: str | None = None,
                     metric: str = DEFAULT_METRIC, cache: bool = True,
                     samples: int = 2000) -> pd.DataFrame:
    """Curva scaricata da W&B (fallback e unica fonte per le diagnostiche).

    `metric` e' la chiave del catalogo: quella davvero loggata nel progetto la
    decide la fonte. La history di W&B e' campionata, `samples` e' il numero
    massimo di punti: le curve hanno 100-300 punti, ben sotto il tetto, quindi i
    valori sono esatti e `scan_history` darebbe lo stesso risultato 20 volte piu'
    lentamente.
    """
    ensure_dirs()
    project = project or sources.DEFAULT_PROJECT
    key = sources.metric_key(project, metric)
    if key is None:
        return pd.DataFrame(columns=["step", "ret", "ret_std_eps"])
    cache_file = CURVE_DIR / f"{project}__{run_id}__{key.replace('/', '_')}.parquet"
    if cache and cache_file.exists():
        return pd.read_parquet(cache_file)
    import wandb

    api = wandb.Api()
    run = api.run(f"{wandb_path(project)}/{run_id}")
    hist = run.history(keys=["global_step", key], samples=samples, pandas=True)
    if hist is None or hist.empty:
        df = pd.DataFrame(columns=["step", "ret", "ret_std_eps"])
    else:
        df = pd.DataFrame({
            "step": hist["global_step"].astype(float),
            "ret": hist[key].astype(float),
            "ret_std_eps": np.nan,
        }).dropna(subset=["ret"])
    if cache:
        df.to_parquet(cache_file, index=False)
    return df


def load_curve(run_id: str, source: str = "auto", eval_map=None,
               metric: str = DEFAULT_METRIC,
               project: str | None = None) -> pd.DataFrame | None:
    """Curva di un run per la metrica richiesta. source: auto | local | wandb.

    Le metriche di eval si leggono dal .npz locale, tutte le altre da W&B.
    """
    project = project or sources.DEFAULT_PROJECT
    field = LOCAL_FIELDS.get(metric)
    if field and source in ("auto", "local"):
        eval_map = build_eval_map() if eval_map is None else eval_map
        path = eval_map.get(run_id)
        if path and Path(path).exists():
            return curve_from_npz(path, field=field)
    if source == "local":
        return None
    mem_key = (project, run_id, metric)
    cached = _mem_get(mem_key)
    if cached is not None:
        return cached if not cached.empty else None
    try:
        df = curve_from_wandb(run_id, project=project, metric=metric)
    except Exception:
        return None
    _mem_put(mem_key, df)
    return df if not df.empty else None


def _report_missing(index: pd.DataFrame, missing: list[str], metric: str) -> None:
    """Perche' mancano: chiave non mappata per quella fonte, o dati assenti.

    Prima le run senza curva sparivano dalla media con un solo conteggio
    generico; se una fonte intera non logga quella metrica e' un fatto da dire.
    """
    if not missing:
        return
    sub = index[index.run_id.isin(missing)]
    by = (sub["project"] if "project" in sub.columns
          else pd.Series(sources.DEFAULT_PROJECT, index=sub.index))
    reasons: dict[str, list[str]] = {}
    for project, group in sub.groupby(by, dropna=False):
        why = sources.unavailable_reason(project, metric)
        bucket = f"{why}" if why else "nessun dato per questa metrica"
        reasons.setdefault(bucket, []).append(f"{project or '?'} ({len(group)})")
    for why, where in reasons.items():
        print(f"[curves] «{metric}» mancante per {', '.join(where)} run: {why}")


def load_curves(index: pd.DataFrame, source: str = "auto", metric: str = DEFAULT_METRIC,
                verbose: bool = True, workers: int = 8) -> pd.DataFrame:
    """Curve di tutti i run dell'indice, in formato tidy [run_id, step, ret].

    Le metriche locali si leggono in sequenza (sono file piccoli); quelle W&B in
    parallelo, perche' ogni run e' una richiesta di rete.
    """
    eval_map = build_eval_map()
    run_ids = list(index.run_id)
    # Ogni run va cercato nel suo progetto: il progetto delle campagne nuove per
    # le une, quello per-environment per le run del paper.
    projects = (dict(zip(index.run_id, index.project.fillna(sources.DEFAULT_PROJECT)))
                if "project" in index.columns else {})

    def fetch(run_id):
        return run_id, load_curve(run_id, source=source, eval_map=eval_map, metric=metric,
                                  project=projects.get(run_id))

    # Le run del paper non hanno .npz: anche per le metriche di eval servono
    # richieste di rete, quindi conviene comunque il pool.
    is_local = (metric in LOCAL_FIELDS and source != "wandb"
                and all(eval_map.get(r) for r in run_ids))
    if is_local or workers <= 1:
        pairs = [fetch(r) for r in run_ids]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pairs = list(pool.map(fetch, run_ids))

    frames, missing = [], []
    for run_id, c in pairs:
        if c is None or c.empty:
            missing.append(run_id)
            continue
        frames.append(c.assign(run_id=run_id))
    if verbose:
        _report_missing(index, missing, metric)
    if not frames:
        return pd.DataFrame(columns=["run_id", "step", "ret"])
    return pd.concat(frames, ignore_index=True)


# --- aggregazione sui seed --------------------------------------------------

def _smooth(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return y
    return pd.Series(y).rolling(window, min_periods=1).mean().to_numpy()


def aggregate(curves: pd.DataFrame, meta: pd.DataFrame, group_cols,
              band: str = "se", smooth: int = 1, grid_points: int | None = None,
              xmax: float | None = None) -> pd.DataFrame:
    """Media sui seed con banda di incertezza.

    Ogni run viene interpolato su una griglia comune al gruppo (i setting con
    n_steps diversi hanno step di eval diversi), poi lisciato con media mobile.

    band: se | std | ci95 | iqr | minmax
    Ritorna [*group_cols, step, mean, lo, hi, n_seeds].
    """
    group_cols = list(group_cols)
    df = curves.merge(meta[["run_id", *group_cols]], on="run_id", how="inner")
    out = []
    for key, g in df.groupby(group_cols, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        runs = list(g.groupby("run_id"))
        t_end = min(r["step"].max() for _, r in runs)
        if xmax is not None:
            t_end = min(t_end, xmax)
        n_pts = grid_points or int(np.median([len(r) for _, r in runs]))
        n_pts = max(int(n_pts), 2)
        grid = np.linspace(0, t_end, n_pts)
        mat = np.vstack([
            _smooth(np.interp(grid, r["step"].to_numpy(), r["ret"].to_numpy()), smooth)
            for _, r in runs
        ])
        mean = mat.mean(axis=0)
        n = mat.shape[0]
        if band == "std":
            half = mat.std(axis=0, ddof=1) if n > 1 else np.zeros_like(mean)
            lo, hi = mean - half, mean + half
        elif band == "ci95":
            se = mat.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
            lo, hi = mean - 1.96 * se, mean + 1.96 * se
        elif band == "iqr":
            lo, hi = np.percentile(mat, 25, axis=0), np.percentile(mat, 75, axis=0)
        elif band == "minmax":
            lo, hi = mat.min(axis=0), mat.max(axis=0)
        elif band == "none":
            lo = hi = mean
        else:  # 'se' (default)
            se = mat.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
            lo, hi = mean - se, mean + se
        block = pd.DataFrame({"step": grid, "mean": mean, "lo": lo, "hi": hi, "n_seeds": n})
        for col, val in zip(group_cols, key):
            block[col] = val
        out.append(block)
    if not out:
        return pd.DataFrame(columns=[*group_cols, "step", "mean", "lo", "hi", "n_seeds"])
    return pd.concat(out, ignore_index=True)


def final_performance(curves: pd.DataFrame, meta: pd.DataFrame, group_cols,
                      last_frac: float = 0.1) -> pd.DataFrame:
    """Prestazione finale per run (media dell'ultimo `last_frac` della curva),
    poi media e deviazione standard sui seed."""
    group_cols = list(group_cols)
    df = curves.merge(meta[["run_id", *group_cols]], on="run_id", how="inner")
    per_run = []
    for run_id, g in df.groupby("run_id"):
        g = g.sort_values("step")
        k = max(1, int(len(g) * last_frac))
        row = {c: g.iloc[-1][c] for c in group_cols}
        row.update(run_id=run_id, final=g["ret"].to_numpy()[-k:].mean())
        per_run.append(row)
    per_run = pd.DataFrame(per_run)
    agg = per_run.groupby(group_cols, dropna=False)["final"].agg(
        mean="mean", std="std", n="size").reset_index()
    agg["se"] = agg["std"] / np.sqrt(agg["n"].clip(lower=1))
    return agg, per_run
