"""Metriche riassuntive di una curva di eval: AUC, prestazione finale, instabilita'.

Sono funzioni della sola curva di eval, quindi si calcolano a posteriori dai
`evaluations.npz` gia' in storage: valgono anche per le campagne gia' chiuse e la
definizione si puo' cambiare senza rilanciare niente.

Tutte e tre si calcolano **per run** e poi si mediano sui seed (mai il contrario:
la media delle curve nasconde la varianza fra i seed, che e' meta' del risultato).

Le run di uno stesso gruppo vengono prima troncate all'orizzonte comune
(`t_end = min` dei loro ultimi step): confrontare l'AUC di una run da 1M step con
una da 3M non vuol dire niente.

  auc          media della curva pesata sui timestep (trapezio / orizzonte), cioe'
               il ritorno medio lungo tutto il training: premia chi sale prima.
               Sta nelle unita' del ritorno, quindi si confronta a parita' di env.
  final        media degli ultimi `last_n` punti di eval (default 10). Attenzione:
               "ultimi 10 punti" dipende da eval_freq, quindi confronta solo run
               con lo stesso eval_freq; con eval_freq diversi usa `last_frac`
               (curves.final_performance) che ragiona in frazione di curva.
  instability  RMSE fra la curva e la sua versione lisciata (media mobile centrata
               di ampiezza `smooth_frac` dell'orizzonte, in timestep e non in punti
               proprio per non dipendere da eval_freq). Misura quanto la curva
               oscilla attorno al proprio andamento, non quanto e' alta.
               - `instability_rel` = RMSE / |media della curva|: adimensionale,
                 e' quella da usare per confrontare env diversi;
               - `instability_net` toglie il rumore dello stimatore di eval
                 (varianza fra i 50 episodi / n_episodi): quel che resta e'
                 oscillazione vera della policy. Richiede il .npz locale
                 (`ret_std_eps`, `n_eps`), altrimenti resta NaN.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

METRIC_COLUMNS = ["auc", "final", "instability", "instability_rel", "instability_net"]


def _smooth_centered(y: np.ndarray, window_points: int) -> np.ndarray:
    """Media mobile centrata, NaN dove la finestra non e' piena.

    Centrata e non trascinata: una media trascinata e' in ritardo sulla curva, e
    il ritardo finirebbe nell'RMSE come se fosse instabilita'. Ai bordi la
    finestra e' per forza asimmetrica e sposta la media dalla curva anche se
    questa e' perfettamente liscia: quei punti si scartano (NaN) invece di
    contarli come oscillazione.
    """
    if window_points <= 1:
        return y
    return pd.Series(y).rolling(window_points, center=True).mean().to_numpy()


def _window_points(step: np.ndarray, smooth_frac: float) -> int:
    """Ampiezza della finestra in punti, partendo da una frazione dell'orizzonte.

    L'ampiezza e' definita in timestep e convertita con la spaziatura mediana fra
    le eval del run: cosi' due run con eval_freq diversi vengono lisciate sulla
    stessa scala temporale.
    """
    if len(step) < 3 or smooth_frac <= 0:
        return 1
    spacing = float(np.median(np.diff(step)))
    if spacing <= 0:
        return 1
    span = smooth_frac * (step[-1] - step[0])
    # al massimo meta' curva: oltre, non resterebbe nessun punto con la finestra piena
    points = int(np.clip(round(span / spacing), 1, max(1, len(step) // 2)))
    # dispari: con una finestra pari il centro cade fra due campioni e la media
    # mobile resta sfasata di mezzo passo, che su una curva in salita e' un
    # residuo costante (pendenza x mezzo passo) scambiato per instabilita'
    return points if points % 2 else points + 1


def curve_metrics(step: np.ndarray, ret: np.ndarray, last_n: int = 10,
                  smooth_frac: float = 0.1,
                  ret_std_eps: np.ndarray | None = None,
                  n_eps: np.ndarray | None = None) -> dict:
    """Le tre metriche per una singola curva gia' ordinata e troncata."""
    if len(step) == 0:
        return {c: np.nan for c in METRIC_COLUMNS}
    if len(step) == 1:
        return {**{c: np.nan for c in METRIC_COLUMNS}, "auc": float(ret[0]),
                "final": float(ret[0])}

    horizon = step[-1] - step[0]
    auc = float(np.trapezoid(ret, step) / horizon) if horizon > 0 else float(ret.mean())

    k = max(1, min(int(last_n), len(ret)))
    final = float(ret[-k:].mean())

    residual = ret - _smooth_centered(ret, _window_points(step, smooth_frac))
    inner = np.isfinite(residual)  # i bordi, dove la finestra non e' piena, restano fuori
    mse = float(np.mean(residual[inner] ** 2)) if inner.any() else np.nan
    instability = float(np.sqrt(mse)) if inner.any() else np.nan
    scale = abs(float(np.mean(ret)))
    instability_rel = instability / scale if scale > 0 else np.nan

    # rumore dello stimatore: var fra gli episodi / n_episodi, mediata sugli stessi punti
    instability_net = np.nan
    if ret_std_eps is not None and n_eps is not None and inner.any():
        noise = np.asarray(ret_std_eps, dtype=float) ** 2 / np.asarray(n_eps, dtype=float)
        if np.isfinite(noise[inner]).all():
            instability_net = float(np.sqrt(max(0.0, mse - float(np.mean(noise[inner])))))

    return dict(auc=auc, final=final, instability=instability,
                instability_rel=instability_rel, instability_net=instability_net)


def per_run_metrics(curves: pd.DataFrame, meta: pd.DataFrame, group_cols,
                    last_n: int = 10, smooth_frac: float = 0.1,
                    xmax: float | None = None) -> pd.DataFrame:
    """Una riga per run con [*group_cols, run_id, *METRIC_COLUMNS, t_end, n_points].

    L'orizzonte comune si calcola dentro ogni gruppo, come in curves.aggregate().
    """
    group_cols = list(group_cols)
    df = curves.merge(meta[["run_id", *group_cols]], on="run_id", how="inner")
    rows = []
    for key, g in df.groupby(group_cols, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        runs = list(g.groupby("run_id"))
        t_end = min(r["step"].max() for _, r in runs)
        if xmax is not None:
            t_end = min(t_end, xmax)
        for run_id, r in runs:
            r = r.sort_values("step")
            r = r[r["step"] <= t_end]
            row = dict(zip(group_cols, key))
            row.update(run_id=run_id, t_end=float(t_end), n_points=len(r))
            row.update(curve_metrics(
                r["step"].to_numpy(dtype=float), r["ret"].to_numpy(dtype=float),
                last_n=last_n, smooth_frac=smooth_frac,
                ret_std_eps=r["ret_std_eps"].to_numpy() if "ret_std_eps" in r else None,
                n_eps=r["n_eps"].to_numpy() if "n_eps" in r else None,
            ))
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=[*group_cols, "run_id", "t_end", "n_points",
                                     *METRIC_COLUMNS])
    return pd.DataFrame(rows)


def summarize(curves: pd.DataFrame, meta: pd.DataFrame, group_cols,
              last_n: int = 10, smooth_frac: float = 0.1,
              xmax: float | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(aggregato sui seed, per run).

    L'aggregato ha, per ogni metrica, `<metrica>` (media sui seed), `<metrica>_std`
    e `<metrica>_se`, piu' `n_seeds`.
    """
    group_cols = list(group_cols)
    per_run = per_run_metrics(curves, meta, group_cols, last_n=last_n,
                              smooth_frac=smooth_frac, xmax=xmax)
    if per_run.empty:
        cols = [*group_cols, "n_seeds"]
        cols += [f"{m}{s}" for m in METRIC_COLUMNS for s in ("", "_std", "_se")]
        return pd.DataFrame(columns=cols), per_run

    grouped = per_run.groupby(group_cols, dropna=False)
    agg = grouped[METRIC_COLUMNS].agg(["mean", "std", "size"])
    out = pd.DataFrame(index=agg.index)
    for m in METRIC_COLUMNS:
        n = agg[(m, "size")]
        out[m] = agg[(m, "mean")]
        out[f"{m}_std"] = agg[(m, "std")]
        out[f"{m}_se"] = agg[(m, "std")] / np.sqrt(n.clip(lower=1))
    out["n_seeds"] = grouped["run_id"].nunique()
    return out.reset_index(), per_run
