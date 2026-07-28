"""Lettura di `plots/style.toml`: le regole scritte a mano che governano i grafici.

Un file solo, dichiarativo, riletto quando cambia (mtime): si salva e la figura
successiva — anteprima o `.tex` — e' gia' quella nuova, senza riavviare niente.

Qui dentro non ci sono default nascosti: quelli veri stanno nel `.toml`, che li
elenca per esteso. `get()` restituisce comunque un valore di scorta se una chiave
e' stata cancellata, cosi' un file monco non fa esplodere il selettore.

Le regole `[[series]]` sono una lista ordinata: vince la prima che combacia. Non
si prova a indovinare la piu' specifica — l'ordine e' scritto nel file, quindi e'
prevedibile e si cambia spostando un blocco.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

from .paths import PLOTS_ROOT

RULES_FILE = PLOTS_ROOT / "style.toml"

# Scorte: servono solo se una chiave sparisce dal .toml, che invece le elenca
# tutte. Tenerle qui evita che una riga cancellata per sbaglio rompa la pagina.
FALLBACK = {
    "figure": {"panel_size": [3.1, 2.3],
               "xlabel": r"Environment Steps ($\times 10^6$)",
               "ylabel": "Mean Return", "xscale": 1e6, "share": "row",
               "font_scale": 1.0},
    "lines": {"width": 1.4, "band_alpha": 0.18, "band": "se", "smooth": 5,
              "baseline_color": "#000000", "baseline_width": 1.6},
    "legend": {"where": "panel", "loc": "best", "ncol": 1, "frame": True,
               "font_size": 8.5},
    "palette": {"colors": ["#648FFF", "#FE6100", "#DC267F", "#785EF0", "#FFB000"]},
    "latex": {"axis_options": [], "preamble": "", "macros": []},
}

_cache: dict | None = None
_stamp: tuple | None = None


def load(force: bool = False) -> dict:
    """Il file, riletto se e' cambiato sul disco."""
    global _cache, _stamp
    try:
        stat = RULES_FILE.stat()
        stamp = (stat.st_mtime_ns, stat.st_size)
    except OSError:
        stamp = None
    if _cache is not None and stamp == _stamp and not force:
        return _cache
    data = {}
    if stamp is not None:
        try:
            data = tomllib.loads(RULES_FILE.read_text())
        except tomllib.TOMLDecodeError as exc:
            # meglio lo stile di default che una pagina bianca: l'errore si legge
            # nel terminale del selettore e si corregge senza perdere la selezione
            print(f"[rules] {RULES_FILE} non e' TOML valido ({exc}): uso i default")
    _cache, _stamp = data, stamp
    return data


def get(section: str, key: str, default=None):
    """Un valore del file, con la scorta di FALLBACK se manca."""
    value = (load().get(section) or {}).get(key)
    if value is not None:
        return value
    if default is not None:
        return default
    return FALLBACK.get(section, {}).get(key)


def series_rules() -> list[dict]:
    return list(load().get("series") or [])


def rule_for(row) -> dict:
    """La prima regola `[[series]]` che combacia con la serie (o {}).

    `row` e' un dizionario con le colonne dell'indice. Un `match` su una colonna
    che la serie non ha non combacia mai: le regole si scrivono sulle dimensioni
    che quella figura distingue davvero.
    """
    for rule in series_rules():
        wanted = rule.get("match") or {}
        if not wanted:
            continue
        if all(_same(row.get(col), value) for col, value in wanted.items()):
            return rule
    return {}


def _same(actual, wanted) -> bool:
    """Confronto tollerante: nell'indice window e setting sono float (4.0 == 4)."""
    if actual is None:
        return False
    if isinstance(wanted, bool) or isinstance(actual, bool):
        return bool(actual) is bool(wanted)
    try:
        return float(actual) == float(wanted)
    except (TypeError, ValueError):
        return str(actual) == str(wanted)


def palette() -> list[str]:
    return list(get("palette", "colors"))


def latex_macros_comment() -> str:
    """Le macro dei nomi, come commento in cima ai .tex esportati."""
    macros = get("latex", "macros") or []
    preamble = get("latex", "preamble") or ""
    lines = []
    if preamble:
        lines.append(f"% {preamble}")
    lines += [f"% {m}" for m in macros]
    return "\n".join(lines)
