"""Selezione e raggruppamento dei run a partire dall'indice.

Sintassi dei filtri da riga di comando (uno per argomento):
    env=Hopper-v5           uguale
    env=Hopper,Reacher      uno di questi (i suffissi -v5 sono opzionali)
    setting!=1              diverso
    window>=4               confronto numerico (>, >=, <, <=)
    opc=false               booleani: true/false
"""
from __future__ import annotations

import re

import pandas as pd

_OPS = ["!=", ">=", "<=", "=", ">", "<"]
ENV_SUFFIX = "-v5"
KNOWN_ENVS = ["Ant", "HalfCheetah", "Hopper", "Reacher", "Swimmer", "Walker2d"]


def _cast(value: str):
    v = value.strip()
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null", "nan", ""):
        return None
    if re.fullmatch(r"-?\d+", v):
        return int(v)
    if re.fullmatch(r"-?\d*\.\d+([eE][-+]?\d+)?", v):
        return float(v)
    if v in KNOWN_ENVS:
        return v + ENV_SUFFIX
    return v


def parse_filter(expr: str):
    for op in _OPS:
        if op in expr:
            key, raw = expr.split(op, 1)
            values = [_cast(x) for x in raw.split(",")] if op in ("=", "!=") else _cast(raw)
            return key.strip(), op, values
    raise ValueError(f"Filtro non valido: {expr!r} (usa key=value, key!=value, key>=value)")


def apply_filters(df: pd.DataFrame, exprs) -> pd.DataFrame:
    """Applica in AND tutti i filtri passati."""
    out = df
    for expr in exprs or []:
        key, op, values = parse_filter(expr)
        if key not in out.columns:
            raise KeyError(f"Colonna sconosciuta: {key}. Disponibili: {sorted(out.columns)}")
        col = out[key]
        if op == "=":
            mask = col.isin(values) if None not in values else (col.isin(values) | col.isna())
        elif op == "!=":
            mask = ~col.isin(values)
        elif op == ">":
            mask = col > values
        elif op == ">=":
            mask = col >= values
        elif op == "<":
            mask = col < values
        else:
            mask = col <= values
        out = out[mask.fillna(False)]
    return out


def select_runs(df: pd.DataFrame, filters=None, state: str | None = "finished",
                dropna_cols=()) -> pd.DataFrame:
    """Filtra l'indice; per default tiene solo i run `finished`."""
    out = df
    if state and state != "any":
        out = out[out.state.isin(state.split(","))]
    out = apply_filters(out, filters)
    for c in dropna_cols:
        out = out[out[c].notna()]
    return out.copy()


def coverage(df: pd.DataFrame, by) -> pd.DataFrame:
    """Numero di seed per combinazione: utile per capire cosa e' disponibile."""
    by = list(by)
    g = df.groupby(by, dropna=False).agg(
        n_seeds=("seed", "nunique"), n_runs=("run_id", "size"),
        seeds=("seed", lambda s: ",".join(str(int(x)) for x in sorted(s.dropna().unique()))),
    )
    return g.reset_index().sort_values(by)
