"""Schema dei campi dell'indice: un'unica dichiarazione per colonna.

Prima di questo modulo le stesse colonne erano elencate a mano in cinque posti
(le dimensioni della sidebar, i campi ammessi su righe/colonne, le dimensioni
che separano le serie, le etichette HTML del selettore, i titoli dei pannelli).
Le liste si erano gia' disallineate: `epoch_mult` mancava fra le dimensioni di
serie, quindi le baseline PPO a epoche moltiplicate finivano mediate insieme.

Qui ogni colonna e' dichiarata una volta con:
  - `title`   etichetta leggibile (sidebar, intestazioni della copertura);
  - `ui`      compare fra i filtri del selettore;
  - `grid`    puo' finire su righe/colonne/colori;
  - `series`  e' una dimensione di ablation: se varia deve separare le curve
              (auto_hue) e va segnalata se non lo fa (warn_merged);
  - `html`    come si scrive il valore in pagina (HTML, niente mathtext);
  - `title_of` titolo del pannello (mathtext di matplotlib);
  - `legend`  frammento che il valore aggiunge alla legenda (None = niente).

L'ordine di dichiarazione e' quello della sidebar ed e' anche l'ordine in cui le
dimensioni compaiono in legenda.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

SETTING_NAMES = {
    1: "Fixed batch size",
    2: r"Fixed $N_\mathcal{B}$",
    3: r"$H/\omega$ steps",
}
SETTING_NAMES_HTML = {
    1: "Fixed batch size",
    2: "Fixed N<sub>B</sub>",
    3: "H/ω steps",
}
SETTING_CAPTIONS = {
    1: r"Fixed batch size setting.",
    2: r"Fixed $N_\mathcal{B}$ setting.",
    3: r"$H/\omega$ steps setting.",
}


def _int(value):
    """Setting/window/epoch_mult vivono nell'indice come float: 4.0 -> 4."""
    return int(float(value))


def _missing(value) -> bool:
    return value is None or (isinstance(value, float) and value != value)


@dataclass(frozen=True)
class Field:
    col: str
    title: str
    ui: bool = False
    grid: bool = False
    series: bool = False
    html: Callable | None = None
    title_of: Callable | None = None
    legend: Callable | None = None


def _bool_html(yes: str = "sì", no: str = "no"):
    return lambda v: yes if v in (True, "True") else no


FIELDS: list[Field] = [
    Field(
        "env", "Environment", ui=True, grid=True, series=True,
        html=lambda v: str(v).replace("-v5", ""),
        title_of=lambda v: str(v).replace("-v5", ""),
        legend=lambda v: str(v),
    ),
    Field("family", "Famiglia", ui=True, grid=True, series=True,
          html=str, title_of=str),
    Field(
        "window", "Window length ω", ui=True, grid=True, series=True,
        html=lambda v: f"ω = {_int(v)}",
        title_of=lambda v: rf"$\omega = {_int(v)}$",
        # omega=1 vuol dire "nessun riuso": in legenda non aggiunge nulla
        legend=lambda v: rf"$\omega = {_int(v)}$" if _int(v) > 1 else None,
    ),
    Field(
        "setting", "Setting", ui=True, grid=True, series=True,
        html=lambda v: f"{_int(v)} · {SETTING_NAMES_HTML.get(_int(v), v)}",
        title_of=lambda v: SETTING_NAMES.get(_int(v), str(v)),
        legend=lambda v: SETTING_NAMES.get(_int(v), str(v)),
    ),
    Field(
        "is_type", "IS", ui=True, grid=True, series=True,
        html=str, title_of=lambda v: f"IS = {v}",
        # con i nomi del paper l'IS e' gia' dentro il nome (ωPPO-U / ωPPO-BH):
        # series_label lo aggiunge solo con --raw-names
        legend=lambda v: f"IS={v}",
    ),
    Field(
        "opc", "Critic on-policy", ui=True, grid=True, series=True,
        html=_bool_html(),
        title_of=lambda v: "on-policy critic" if v else "off-policy critic",
        # in legenda l'off-policy critic e' il suffisso "Off" attaccato al nome,
        # non un elemento a se': lo gestisce series_label
        legend=None,
    ),
    Field(
        "fresh_adv", "Fresh advantages", ui=True, grid=True, series=True,
        html=_bool_html(), title_of=lambda v: f"fresh_adv = {bool(v)}",
        legend=lambda v: "fresh" if v else "stale",
    ),
    Field(
        "adaptive_lr", "Adaptive LR", ui=True, grid=True, series=True,
        html=_bool_html(), title_of=lambda v: f"adaptive_lr = {bool(v)}",
        legend=lambda v: "adaLR" if v else "fixLR",
    ),
    Field(
        # tre valori (random | balanced | weighted), come batch_sampling nei
        # config: il vecchio booleano balanced_batches e' mappato qui.
        "sampling", "Batch sampling", ui=True, grid=True, series=True,
        html=str, title_of=lambda v: f"sampling = {v}", legend=str,
    ),
    Field(
        "epoch_mult", "Epoche (× base)", ui=True, grid=True, series=True,
        html=lambda v: f"×{_int(v)}",
        title_of=lambda v: rf"$\times {_int(v)}$ epochs",
        legend=lambda v: rf"$\times {_int(v)}$ epochs",
    ),
    Field("ablation", "Ablation extra", ui=True),
    Field("total_timesteps", "Orizzonte", ui=True,
          html=lambda v: f"{float(v) / 1e6:g}M step"),
    Field("state", "Stato", ui=True),
    Field("campaign", "Campagna", ui=True),
    Field("project", "Progetto W&B", ui=True),
    # non filtrabile dalla sidebar, ma utilizzabile in legenda
    Field("seed", "Seed", legend=lambda v: f"seed={_int(v)}"),
]

BY_COL: dict[str, Field] = {f.col: f for f in FIELDS}

UI_DIMENSIONS = [f.col for f in FIELDS if f.ui]
GRID_FIELDS = [f.col for f in FIELDS if f.grid]
# Dimensioni che devono separare le curve: se una varia e nessuno l'ha messa su
# colori/righe/colonne, configurazioni diverse finiscono mediate insieme.
SERIES_FIELDS = [f.col for f in FIELDS if f.series]


def title(col: str) -> str:
    f = BY_COL.get(col)
    return f.title if f else col


def html_value(col: str, value) -> str:
    """Valore come va scritto nella pagina del selettore."""
    f = BY_COL.get(col)
    if f is not None and f.html is not None:
        try:
            return f.html(value)
        except (TypeError, ValueError, KeyError):
            pass
    # Nota: in Python 1.0 == True, quindi il ramo booleano deve venire dopo i
    # formattatori dichiarati, altrimenti ogni valore pari a 1 diventa "sì".
    if _missing(value):
        return "—"
    if value in (True, "True"):
        return "sì"
    if value in (False, "False"):
        return "no"
    return str(value)


def panel_title(col: str, value, paper: bool = True) -> str:
    """Titolo di riga/colonna della griglia (mathtext)."""
    if _missing(value):
        return ""
    f = BY_COL.get(col)
    if f is not None and f.title_of is not None:
        try:
            return f.title_of(value)
        except (TypeError, ValueError, KeyError):
            pass
    return f"{col} = {value}"


def legend_bit(col: str, value) -> str | None:
    """Frammento che il campo aggiunge alla legenda, o None."""
    if _missing(value):
        return None
    f = BY_COL.get(col)
    if f is None or f.legend is None:
        return None
    try:
        return f.legend(value)
    except (TypeError, ValueError, KeyError):
        return None
