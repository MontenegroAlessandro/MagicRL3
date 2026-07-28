"""Selezioni salvate: lettura, scrittura, migrazione dal formato vecchio.

Una selezione e' quello che il selettore ha in mano quando premi «Salva»: le
run scelte, i filtri con cui ci sei arrivato e — la parte che conta per gli
script — lo `FigureSpec` con cui la pagina stava disegnando. Salvare e' quindi
salvare la figura, non solo l'elenco delle run: `plot_curves.py --runs-file`
rifa' esattamente quella figura, baseline compresa.

`selection.json` e' sempre l'ultima salvata (quella che gli script usano senza
argomenti); `selections/<slug>.json` e' lo storico per nome.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .figure import SPEC_VERSION, FigureSpec
from .paths import SELECTION_JSON, SELECTIONS_DIR


def slugify(name: str) -> str:
    slug = "".join(c if c.isalnum() else "-" for c in name.strip().lower())
    slug = "-".join(p for p in slug.split("-") if p)[:60]
    return slug or datetime.now().strftime("selezione-%Y%m%d-%H%M%S")


# --- migrazione -------------------------------------------------------------

# La colonna booleana `balanced` e' diventata `sampling` a tre valori
# (random | balanced | weighted), come batch_sampling nei config.
_BALANCED_TO_SAMPLING = {"True": "balanced", "False": "random",
                         "true": "balanced", "false": "random"}


def _migrate(data: dict) -> dict:
    """Porta al formato corrente una selezione salvata con la versione 1.

    Nella versione 1 le impostazioni di disegno stavano in `grid` e non
    comprendevano ne' le etichette ne' le opzioni di griglia: quel che c'era
    diventa uno `FigureSpec`, il resto prende i default.
    """
    if data.get("version") == SPEC_VERSION and "spec" in data:
        return data
    grid = data.get("grid") or {}
    baseline = grid.get("baseline") or {}
    spec = {
        "run_ids": data.get("run_ids"),
        "state": "any",                    # lo stato l'aveva gia' scelto la UI
        "metric": grid.get("metric"),
        "rows": grid.get("rows") or None,
        "cols": grid.get("cols") or None,
        "hue": ["sampling" if h == "balanced" else h
                for h in (grid.get("hue") or [])] or None,
        "band": grid.get("band", "se"),
        "smooth": grid.get("smooth", 5),
        "baseline": {"family": baseline.get("family") or None,
                     "epochs": baseline.get("epochs") or ""},
        "row_captions": "auto" if grid.get("rows") else "off",
    }
    spec = {k: v for k, v in spec.items() if v is not None}
    dims = dict(data.get("dims") or {})
    if "balanced" in dims:
        # nella versione 1 i dims erano liste nude, senza operatore
        old_values = dims.pop("balanced")
        if isinstance(old_values, list):
            dims["sampling"] = [_BALANCED_TO_SAMPLING.get(str(v), str(v))
                                for v in old_values]
    out = dict(data)
    out.update(version=SPEC_VERSION, spec=spec, dims=dims)
    out.pop("grid", None)
    return out


def read(path: str | Path) -> dict:
    """Selezione da file, migrata al formato corrente."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Selezione non trovata: {path} (salvala dal selettore)")
    return _migrate(json.loads(path.read_text()))


def spec_from(path: str | Path) -> tuple[FigureSpec, dict]:
    """(FigureSpec, selezione) da un file: quello che serve agli script."""
    data = read(path)
    spec = FigureSpec.from_dict(data.get("spec") or {})
    if not spec.run_ids:
        spec.run_ids = data.get("run_ids")
    return spec, data


# --- storico ----------------------------------------------------------------

def path_for(slug: str) -> Path:
    return SELECTIONS_DIR / f"{slugify(slug)}.json"


def listing() -> list[dict]:
    """Storico delle selezioni salvate, dalla piu' recente."""
    items = []
    for path in SELECTIONS_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        items.append({
            "slug": path.stem,
            "name": data.get("name") or path.stem,
            "saved_at": data.get("saved_at", ""),
            "n_runs": data.get("n_runs", 0),
            "summary": " ".join(data.get("filter_args") or []) or "nessun filtro",
            "path": str(path),
        })
    return sorted(items, key=lambda d: d["saved_at"], reverse=True)


def write(payload: dict) -> Path:
    """Salva la selezione e la rende quella attiva (`selection.json`)."""
    SELECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    stored = path_for(payload["slug"])
    text = json.dumps(payload, indent=1, default=str)
    stored.write_text(text)
    SELECTION_JSON.write_text(text)
    return stored


def activate(slug: str) -> dict:
    """Rende attiva una selezione dello storico e la restituisce."""
    data = read(path_for(slug))
    SELECTION_JSON.write_text(json.dumps(data, indent=1, default=str))
    return data


def rename(slug: str, name: str) -> dict:
    """Cambia solo l'etichetta: slug e nome del file restano quelli di partenza."""
    stored = path_for(slug)
    data = read(stored)
    data["name"] = name
    stored.write_text(json.dumps(data, indent=1, default=str))
    if SELECTION_JSON.exists():
        try:
            active = json.loads(SELECTION_JSON.read_text())
        except json.JSONDecodeError:
            active = {}
        if active.get("slug") == slugify(slug):
            SELECTION_JSON.write_text(json.dumps(data, indent=1, default=str))
    return data


def delete(slug: str) -> None:
    path_for(slug).unlink(missing_ok=True)
