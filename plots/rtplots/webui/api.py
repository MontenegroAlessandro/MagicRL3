"""Handler del selettore: funzioni pure (indice + richiesta -> dizionario).

Nessuna di queste funzioni sa che esiste HTTP: si provano da sole, e il server
(`server.py`) si limita a instradare. Il vocabolario delle dimensioni non e'
ridefinito qui — viene da `schema.py`, lo stesso che scrive i titoli dei
pannelli e le voci di legenda.
"""
from __future__ import annotations

import io
import time
from datetime import datetime

import matplotlib
import pandas as pd

from .. import figure as F
from .. import rules as R
from .. import schema, selection, style as S, tikz
from ..metrics import DEFAULT_METRIC, metric_info, ui_groups
from ..paths import SELECTION_JSON

# Metriche W&B: ogni run non in cache e' una richiesta di rete. Oltre la soglia
# si preferisce dirlo invece di far aspettare minuti.
MAX_WANDB_RUNS = 120
MAX_PREVIEW_RUNS = 800
MAX_PANELS = 24


# --- vocabolario delle dimensioni -------------------------------------------

def dimension_values(df: pd.DataFrame) -> list[dict]:
    out = []
    for col in schema.UI_DIMENSIONS:
        if col not in df.columns:
            continue
        vals = df[col].dropna().unique().tolist()
        if not vals:
            continue
        try:
            vals = sorted(vals)
        except TypeError:
            vals = sorted(vals, key=str)
        out.append({
            "col": col,
            "title": schema.title(col),
            "values": [{"value": str(v), "label": schema.html_value(col, v),
                        "count": int((df[col].astype(str) == str(v)).sum())} for v in vals],
        })
    return out


# --- filtri della UI --------------------------------------------------------

# Operatori scegliibili per ogni dimensione. `is`/`in` tengono i valori scelti,
# `is_not`/`not_in` li escludono; la differenza fra i due di ogni coppia e' solo
# quanti valori si possono selezionare, e vale nella pagina (uno solo contro
# molti). Qui i comportamenti sono due: tieni oppure escludi.
OPS = ("is", "is_not", "in", "not_in")
NEGATIVE_OPS = ("is_not", "not_in")
SINGLE_OPS = ("is", "is_not")
DEFAULT_OP = "in"


def dim_filter(raw) -> tuple[str, list]:
    """Filtro di una dimensione -> (operatore, valori).

    Accetta sia la forma corrente `{"op": ..., "values": [...]}` sia la lista
    nuda delle selezioni salvate prima che gli operatori esistessero.
    """
    if isinstance(raw, dict):
        op = raw.get("op") or DEFAULT_OP
        return (op if op in OPS else DEFAULT_OP), list(raw.get("values") or [])
    return DEFAULT_OP, list(raw or [])


def apply_ui_filters(df: pd.DataFrame, sel: dict, exclusions: bool = True) -> pd.DataFrame:
    """Filtri della UI: nessun valore scelto = dimensione non filtrata.

    `exclusions=False` ignora le run escluse a mano dalla tabella di copertura:
    serve per costruire la tabella stessa, che deve continuare a mostrare anche
    le righe deselezionate (altrimenti sparirebbero e non si potrebbero
    riattivare).
    """
    out = df
    for col, raw in (sel.get("dims") or {}).items():
        op, values = dim_filter(raw)
        if not values or col not in out.columns:
            continue
        hit = out[col].astype(str).isin({str(v) for v in values})
        out = out[~hit] if op in NEGATIVE_OPS else out[hit]
    seeds = sel.get("seeds") or {}
    if seeds.get("min") is not None:
        out = out[out.seed >= float(seeds["min"])]
    if seeds.get("max") is not None:
        out = out[out.seed <= float(seeds["max"])]
    excluded = set(sel.get("excluded") or [])
    if exclusions and excluded:
        out = out[~out.run_id.isin(excluded)]
    return out


def live_counts(df: pd.DataFrame, sel: dict) -> dict:
    """Per ogni valore, quante run resterebbero scegliendolo.

    Il conteggio di una dimensione ignora i filtri della dimensione stessa (come
    nelle ricerche a faccette): cosi' i numeri dicono quanto resta *in piu'* o
    *in meno* rispetto a quello che si sta guardando.

    Con un operatore negativo scegliere un valore vuol dire toglierlo, quindi il
    numero e' quante run resterebbero **escludendolo**: mostrare il conteggio
    delle run che lo hanno direbbe l'opposto di quello che succede cliccando.
    """
    dims = sel.get("dims") or {}
    out = {}
    for col in schema.UI_DIMENSIONS:
        if col not in df.columns:
            continue
        others = {k: v for k, v in dims.items() if k != col}
        sub = apply_ui_filters(df, {**sel, "dims": others})
        counts = sub[col].astype(str).value_counts()
        op, _ = dim_filter(dims.get(col))
        if op in NEGATIVE_OPS:
            # le esclusioni gia' attive restano: il numero risponde a «e se
            # togliessi anche questo?»
            kept = apply_ui_filters(df, {**sel, "dims": {**others, col: dims[col]}})
            base = len(kept)
            kept_counts = kept[col].astype(str).value_counts()
            out[col] = {str(k): int(base - kept_counts.get(str(k), 0))
                        for k in counts.index}
        else:
            out[col] = {str(k): int(v) for k, v in counts.items()}
    return out


def _clean(value) -> str:
    """'1.0' -> '1' (l'indice tiene setting/window come float)."""
    try:
        f = float(value)
        return str(int(f)) if f.is_integer() else str(f)
    except (TypeError, ValueError):
        return str(value)


def filter_args(sel: dict, df: pd.DataFrame) -> list[str]:
    """Filtri della UI tradotti nella sintassi --filter degli script."""
    args = []
    for col, raw in (sel.get("dims") or {}).items():
        op, values = dim_filter(raw)
        if not values or col not in df.columns:
            continue
        negative = op in NEGATIVE_OPS
        if not negative and len(values) == df[col].dropna().astype(str).nunique():
            continue  # tutte le opzioni selezionate: filtro inutile
        # `!=` degli script accetta una lista e vale come "nessuno di questi"
        args.append(f"{col}{'!=' if negative else '='}"
                    + ",".join(_clean(v) for v in values))
    seeds = sel.get("seeds") or {}
    all_seeds = df.seed.dropna()
    if seeds.get("min") is not None and (all_seeds.empty or seeds["min"] > all_seeds.min()):
        args.append(f"seed>={int(seeds['min'])}")
    if seeds.get("max") is not None and (all_seeds.empty or seeds["max"] < all_seeds.max()):
        args.append(f"seed<={int(seeds['max'])}")
    return args


def coverage_rows(sel_df: pd.DataFrame, excluded=(), limit: int = 300) -> dict:
    """Tabella di copertura sulle sole dimensioni che variano nella selezione.

    Ogni riga porta con se' i propri `run_ids`: e' cosi' che la pagina puo'
    spuntare o togliere una combinazione dal grafico senza toccare i filtri.
    `sel_df` deve essere la selezione *prima* delle esclusioni, altrimenti le
    righe tolte sparirebbero dalla tabella.
    """
    varying = [c for c in schema.GRID_FIELDS
               if c in sel_df.columns and sel_df[c].nunique(dropna=False) > 1]
    if not varying:
        varying = [c for c in ("family", "env") if c in sel_df.columns]
    if not varying:
        return {"columns": [], "rows": []}
    excluded = set(excluded)
    g = (sel_df.groupby(varying, dropna=False)
         .agg(n_runs=("run_id", "size"), n_seeds=("seed", "nunique"),
              seeds=("seed", lambda s: ",".join(str(int(x)) for x in sorted(s.dropna().unique()))),
              run_ids=("run_id", list))
         .reset_index())
    try:
        g = g.sort_values(varying)
    except TypeError:
        pass
    rows = []
    for r in g.head(limit).to_dict("records"):
        ids = list(r["run_ids"])
        kept = [i for i in ids if i not in excluded]
        seeds_kept = sel_df[sel_df.run_id.isin(kept)].seed.dropna().unique() if kept else []
        rows.append({
            "cells": [schema.html_value(c, r[c]) for c in varying],
            "n_runs": int(r["n_runs"]), "n_seeds": int(r["n_seeds"]), "seeds": r["seeds"],
            "run_ids": ids,
            # quante restano davvero dopo le esclusioni: la riga resta visibile
            # anche quando e' spenta del tutto
            "n_kept": len(kept), "n_seeds_kept": int(len(set(seeds_kept))),
            "on": len(kept) > 0,
        })
    return {"columns": [schema.title(c) for c in varying], "rows": rows,
            "truncated": len(g) > limit}


# --- dalla richiesta della pagina allo spec ---------------------------------

def spec_from_payload(payload: dict, sub: pd.DataFrame) -> F.FigureSpec:
    """Impostazioni della pagina -> FigureSpec (lo stesso che usa la CLI)."""
    grid = dict(payload.get("grid") or {})
    baseline = grid.pop("baseline", None) or {}
    spec = F.FigureSpec.from_dict({
        **{k: v for k, v in grid.items() if v not in ("", None)},
        "run_ids": list(sub.run_id),
        "state": "any",                    # lo stato l'ha gia' scelto la UI
        "baseline": baseline,
        "series_overrides": payload.get("series_overrides") or {},
        # la pagina non ha (ancora) i controlli per queste: default del paper
        "row_captions": "auto" if grid.get("rows") else "off",
    })
    if not spec.metric:
        spec.metric = DEFAULT_METRIC
    return spec


def wandb_cost_guard(sub: pd.DataFrame, metric: str | None) -> str | None:
    """Le metriche W&B costano una richiesta per run: oltre una soglia si ferma."""
    from ..curves import _MEM  # cache condivisa: quel che c'e' non si ripaga

    info = metric_info(metric or DEFAULT_METRIC)
    if info["source"] != "wandb":
        return None
    projects = dict(zip(sub.run_id, sub.get("project", pd.Series(dtype=str))))
    todo = [r for r in sub.run_id if (projects.get(r), r, info["key"]) not in _MEM]
    if len(todo) > MAX_WANDB_RUNS:
        return (f"«{info['label']}» va scaricata da W&B: {len(todo)} run non ancora in "
                f"cache (massimo {MAX_WANDB_RUNS}). Restringi la selezione; dopo il primo "
                f"scaricamento le curve restano in cache.")
    return None


def render(index: pd.DataFrame, sub: pd.DataFrame, payload: dict,
           fmt: str = "png", dpi: int = 110, plot_lock=None) -> dict:
    """Disegna la figura e la restituisce come byte grezzi.

    Stesso percorso per anteprima (png), download (jpeg) e LaTeX (pdf).
    """
    spec = spec_from_payload(payload, sub)
    try:
        series = F.prepare(index[index.run_id.isin(sub.run_id)], spec,
                           full_index=index, verbose=False)
    except ValueError as exc:
        return {"error": str(exc)}
    panels = F.n_panels(series, spec)
    if panels > MAX_PANELS:
        return {"error": f"Troppi pannelli ({panels}): restringi i filtri o le dimensioni."}

    lock = plot_lock if plot_lock is not None else _NullLock()
    with lock:
        S.apply_style()
        fig = F.draw(series, spec)
        buf = io.BytesIO()
        # jpeg non ha canale alpha: sfondo bianco esplicito
        fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight",
                    facecolor="white" if fmt in ("jpg", "jpeg") else "auto")
        matplotlib.pyplot.close(fig)
    return {"raw": buf.getvalue(),
            "series": int(series.agg.label.nunique()), "panels": int(panels),
            "hue": [schema.title(h) for h in series.hue],
            "auto_hue": not (payload.get("grid") or {}).get("hue"),
            "metric": series.metric_label,
            "merged": [schema.title(c) for c in series.merged],
            "series_list": series_list(series, spec),
            "palette": R.palette(),
            "n_seeds": int(series.agg.n_seeds.max()) if len(series.agg) else 0}


def series_list(series: F.Series, spec: F.FigureSpec) -> list[dict]:
    """Le serie della figura, con quello che serve alla pagina per ritoccarle.

    `key` e' l'etichetta di partenza, quella con cui si indicizza il ritocco:
    resta la stessa anche dopo aver rinominato, altrimenti al secondo giro il
    ritocco non troverebbe piu' la sua serie.
    """
    overrides = spec.series_overrides or {}
    renamed = {(o.get("name") or "").strip() or k: k for k, o in overrides.items()}
    out = []
    for label in series.order:
        key = renamed.get(label, label)
        style = series.styles.get(label) or {}
        out.append({
            "key": key,
            "label": label,
            "color": style.get("color"),
            "renamed": key != label,
            "recolored": bool((overrides.get(key) or {}).get("color")),
            "rule": rule_snippet(label, series.matches.get(label) or {}, style),
        })
    return out


def rule_snippet(label: str, match: dict, style: dict) -> str:
    """Il blocco `[[series]]` da incollare in style.toml per rendere fisso il ritocco."""
    pairs = ", ".join(f"{col} = {_toml_value(v)}" for col, v in match.items()
                      if v is not None and v == v)
    lines = [f"match = {{ {pairs} }}"] if pairs else ["match = { }  # da completare"]
    if style.get("color"):
        lines.append(f'color = "{style["color"]}"')
    lines.append(f"name  = {_toml_literal(label)}")
    if style.get("latex"):
        lines.append(f"latex = {_toml_literal(style['latex'])}")
    return "[[series]]\n" + "\n".join(lines)


def _toml_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, (int, float)):
        return str(value)
    return f'"{value}"'


def _toml_literal(text: str) -> str:
    """Stringa TOML letterale: i backslash di LaTeX non vanno raddoppiati."""
    return f"'{text}'" if "'" not in text else f'"{text}"'


class _NullLock:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# --- nomi e snippet ---------------------------------------------------------

def figure_name(sel_df: pd.DataFrame) -> str:
    """Nome file a partire dalle dimensioni fissate nella selezione."""
    bits = []
    for col, prefix in (("env", ""), ("family", ""), ("setting", "s"),
                        ("window", "w"), ("is_type", "is")):
        if col not in sel_df.columns:
            continue
        vals = sel_df[col].dropna().unique()
        if len(vals) == 1:
            bits.append(prefix + _clean(vals[0]).replace("-v5", "").replace("-", "").lower())
    name = "_".join(bits) or "selezione"
    return "rt_" + "".join(c for c in name if c.isalnum() or c in "_")[:60]


def _caption(sel_df: pd.DataFrame, info: dict, band: str) -> str:
    envs = ", ".join(sorted(sel_df.env.dropna().unique())) or "?"
    fams = ", ".join(sorted(sel_df.family.dropna().unique())) or "?"
    band_txt = {"se": "errore standard", "std": "deviazione standard",
                "ci95": "intervallo di confidenza al 95\\%", "iqr": "intervallo interquartile",
                "minmax": "minimo--massimo", "none": "nessuna banda"}.get(band, band)
    return (f"{info.get('metric', 'Mean return')} su {envs} ({fams}). Media su "
            f"{info.get('n_seeds', '?')} seed, banda: {band_txt}.")


def tex_panels(index: pd.DataFrame, sub: pd.DataFrame, payload: dict,
               name: str, plot_lock=None) -> dict:
    """Un sorgente pgfplots per pannello: {files: [(nome, codice)], latex: ...}.

    Ogni riquadro della griglia diventa un `.tex` a se': i pannelli di una figura
    da paper si compongono in LaTeX, non in matplotlib. Lo snippet restituito e'
    la figura gia' montata, con un `\\input` per pannello.
    """
    err = tikz.unavailable_reason()
    if err:
        return {"error": err}
    spec = spec_from_payload(payload, sub)
    try:
        series = F.prepare(index[index.run_id.isin(sub.run_id)], spec,
                           full_index=index, verbose=False)
    except ValueError as exc:
        return {"error": str(exc)}
    panels = F.split_panels(series, spec)
    if len(panels) > MAX_PANELS:
        return {"error": f"Troppi pannelli ({len(panels)}): restringi i filtri."}

    lock = plot_lock if plot_lock is not None else _NullLock()
    files = []
    with lock:
        S.apply_style()
        for panel in panels:
            fig = F.draw(panel.series, panel.spec)
            header = f"% {name}{' — ' + panel.caption if panel.caption else ''}"
            files.append((f"{name}{'_' + panel.slug if panel.slug else ''}.tex",
                          tikz.figure_to_tex(fig, header=header,
                                             styles=panel.series.styles)))
            matplotlib.pyplot.close(fig)

    grid = payload.get("grid") or {}
    info = {"metric": series.metric_label,
            "n_seeds": int(series.agg.n_seeds.max()) if len(series.agg) else 0}
    ncol = max(1, len({p.col for p in panels}))
    return {"files": files, "n_panels": len(panels), "metric": series.metric_label,
            "latex": tex_snippet(name, panels, [f for f, _ in files], ncol,
                                 _caption(sub, info, grid.get("band", "se")))}


def tex_snippet(name: str, panels, filenames: list[str], ncol: int,
                caption: str) -> str:
    """Figura montata: un `\\input` per pannello, in subfigure se sono piu' di uno."""
    if len(filenames) == 1:
        body = f"  \\input{{figures/{filenames[0]}}}\n"
    else:
        width = f"{0.98 / ncol:.2f}".lstrip("0")
        rows = []
        for panel, filename in zip(panels, filenames):
            lines = [f"  \\begin{{subfigure}}[b]{{{width}\\linewidth}}",
                     f"    \\input{{figures/{filename}}}"]
            if panel.caption:
                lines.append(f"    \\caption{{{panel.caption}}}")
            lines.append("  \\end{subfigure}")
            rows.append("\n".join(lines))
            # a fine riga si va a capo, dentro la riga i pannelli stanno affiancati
            rows.append("  \\\\" if panel.col == ncol - 1 else "  \\hfill")
        body = "\n".join(rows[:-1]) + "\n"
    return (f"% {tikz.PREAMBLE.lstrip('% ')}"
            + (" \\usepackage{subcaption}\n" if len(filenames) > 1 else "\n")
            + "\\begin{figure}[t]\n"
              "  \\centering\n"
            + body
            + f"  \\caption{{{caption}}}\n"
              f"  \\label{{fig:{name}}}\n"
              "\\end{figure}")


def latex_snippet(name: str, sel_df: pd.DataFrame, info: dict, band: str) -> str:
    """Blocco figure pronto da incollare, con caption descrittiva."""
    caption = _caption(sel_df, info, band)
    return ("\\begin{figure}[t]\n"
            "  \\centering\n"
            f"  \\includegraphics[width=\\linewidth]{{figures/{name}.pdf}}\n"
            f"  \\caption{{{caption}}}\n"
            f"  \\label{{fig:{name}}}\n"
            "\\end{figure}")


# --- risposte agli endpoint -------------------------------------------------

# Etichette degli operatori nella pagina (l'ordine e' quello della tendina).
OP_LABELS = [
    {"op": "is", "label": "è", "multi": False},
    {"op": "is_not", "label": "non è", "multi": False},
    {"op": "in", "label": "fra", "multi": True},
    {"op": "not_in", "label": "non fra", "multi": True},
]


def dimensions(df: pd.DataFrame) -> dict:
    seeds = df.seed.dropna()
    return {
        "dimensions": dimension_values(df),
        "ops": OP_LABELS,
        "default_op": DEFAULT_OP,
        "grid_fields": [{"col": c, "title": schema.title(c)}
                        for c in schema.GRID_FIELDS if c in df.columns],
        "seed_min": int(seeds.min()) if len(seeds) else 1,
        "seed_max": int(seeds.max()) if len(seeds) else 10,
        "n_runs": int(len(df)),
        "selection_path": str(SELECTION_JSON),
        "selections": selection.listing(),
        "metrics": ui_groups(),
        "default_metric": DEFAULT_METRIC,
    }


def query(df: pd.DataFrame, payload: dict) -> dict:
    sub = apply_ui_filters(df, payload)
    # la copertura si costruisce prima delle esclusioni, cosi' le righe spente
    # restano in tabella e si possono riattivare
    unfiltered = apply_ui_filters(df, payload, exclusions=False)
    grid_cols = [c for c in schema.GRID_FIELDS if c in sub.columns]
    return {
        "n_runs": int(len(sub)),
        "n_excluded": int(len(unfiltered) - len(sub)),
        "n_configs": int(sub.groupby(grid_cols, dropna=False).ngroups) if len(sub) else 0,
        "states": sub.state.value_counts().to_dict() if len(sub) else {},
        "coverage": (coverage_rows(unfiltered, payload.get("excluded") or [])
                     if len(unfiltered) else {"columns": [], "rows": []}),
        "counts": live_counts(df, payload),
        "filter_args": filter_args(payload, df),
    }


def preview(df: pd.DataFrame, payload: dict, plot_lock=None) -> dict:
    sub = apply_ui_filters(df, payload)
    if sub.empty:
        return {"error": "Nessuna run selezionata."}
    if len(sub) > MAX_PREVIEW_RUNS:
        return {"error": f"{len(sub)} run: troppe per l'anteprima."}
    err = wandb_cost_guard(sub, (payload.get("grid") or {}).get("metric"))
    if err:
        return {"error": err}
    t0 = time.time()
    res = render(df, sub, payload, fmt="png", dpi=110, plot_lock=plot_lock)
    res["elapsed"] = round(time.time() - t0, 2)
    return res


def save(df: pd.DataFrame, payload: dict) -> dict:
    sub = apply_ui_filters(df, payload)
    name = (payload.get("name") or "").strip() or \
        datetime.now().strftime("selezione %d/%m %H:%M")
    slug = selection.slugify(name)
    stored = selection.write({
        "version": F.SPEC_VERSION,
        "name": name,
        "slug": slug,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "n_runs": int(len(sub)),
        "filter_args": filter_args(payload, df),
        "dims": payload.get("dims") or {},
        "series_overrides": payload.get("series_overrides") or {},
        "seeds": payload.get("seeds") or {},
        # run tolte a mano dalla copertura: senza, riaprendo la selezione
        # tornerebbero dentro
        "excluded": list(payload.get("excluded") or []),
        "run_ids": sub.run_id.tolist(),
        "spec": spec_from_payload(payload, sub).to_dict(),
    })
    print(f"[selector] salvata «{name}»: {len(sub)} run -> {stored}")
    return {"ok": True, "path": str(stored), "slug": slug, "name": name,
            "n_runs": int(len(sub)), "items": selection.listing()}
