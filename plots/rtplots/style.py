"""Stile dei grafici: applica le regole di `plots/style.toml` a matplotlib.

I valori non stanno qui ma nel `.toml`, che si modifica a mano (vedi
`rules.py`): questo modulo li traduce in rcParams e in colori delle serie. Le
costanti rimaste sono solo quelle che il file non espone.

Preferenze di partenza (vedi plots/README.md):
  - palette IBM colorblind: blu, viola, magenta, arancione, oro;
  - baseline (PPO/SAC/TD3) sempre nera;
  - font serif con mathtext, nomi degli algoritmi in monospace;
  - pannelli con box completo (4 spine), tick verso l'esterno, niente griglia;
  - curva = media sui seed, banda ombreggiata semitrasparente;
  - asse x in milioni di step: "Environment Steps ($\\times 10^6$)";
  - asse y "Mean Return";
  - legenda dentro il pannello, con cornice.
"""
from __future__ import annotations

import re
from itertools import cycle

import matplotlib as mpl
import matplotlib.pyplot as plt

from . import rules as R

# --- Palette IBM colorblind-safe -------------------------------------------
# Riferimento storico: la palette viva e' [palette].colors in style.toml.
IBM = {
    "blue": "#648FFF",
    "purple": "#785EF0",
    "magenta": "#DC267F",
    "orange": "#FE6100",
    "gold": "#FFB000",
}
# Ordine di assegnazione dei colori alle serie (come nella figura di riferimento:
# blu, arancione, magenta, viola, oro).
IBM_ORDER = [IBM["blue"], IBM["orange"], IBM["magenta"], IBM["purple"], IBM["gold"]]

# Fallback aggiuntivi se le serie superano i 5 colori IBM (varianti di luminosita').
IBM_EXTENDED = IBM_ORDER + ["#1F5AE0", "#B34700", "#8A0F4E", "#4B34C0", "#A87200"]


def band_alpha() -> float:
    return float(R.get("lines", "band_alpha"))


def line_width() -> float:
    return float(R.get("lines", "width"))


def baseline_color() -> str:
    return str(R.get("lines", "baseline_color"))


def baseline_width() -> float:
    return float(R.get("lines", "baseline_width"))


def baseline_styles() -> list[str]:
    """Tratteggi di default delle baseline, nell'ordine in cui vengono assegnati.

    Con piu' di una baseline nello stesso pannello il colore non basta a
    distinguerle (sono nere per convenzione): la prima e' continua, le altre
    seguono questa lista. Stesse parole di `[[series]].style` (solid, dashed,
    dotted, dashdot), cosi' l'anteprima le puo' ritoccare con lo stesso
    controllo. "dashed" resta riservato alla baseline a epoche moltiplicate.
    """
    styles = list(R.get("lines", "baseline_styles") or [])
    return styles or ["solid", "dashdot", "dotted"]


def color_cycle(n: int) -> list[str]:
    """n colori distinti dalla palette del file, allungata se non bastano."""
    palette = R.palette() or IBM_ORDER
    if n <= len(palette):
        return palette[:n]
    it = cycle(palette)
    return [next(it) for _ in range(n)]


def apply_style(scale: float | None = None) -> None:
    """rcParams globali dalle regole. `scale` sovrascrive [figure].font_scale."""
    scale = float(R.get("figure", "font_scale") if scale is None else scale)
    mpl.rcParams.update({
        # font
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "STIX Two Text", "serif"],
        "font.monospace": ["DejaVu Sans Mono", "Courier New", "monospace"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 10 * scale,
        "axes.labelsize": 11 * scale,
        "axes.titlesize": 11 * scale,
        "xtick.labelsize": 9 * scale,
        "ytick.labelsize": 9 * scale,
        "legend.fontsize": float(R.get("legend", "font_size")) * scale,
        # assi: box completo, niente griglia
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "axes.axisbelow": True,
        # tick verso l'esterno
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        # linee
        "lines.linewidth": line_width(),
        "lines.solid_capstyle": "round",
        # legenda con cornice, come nella figura di riferimento
        "legend.frameon": bool(R.get("legend", "frame")),
        "legend.framealpha": 1.0,
        "legend.fancybox": False,
        "legend.edgecolor": "0.3",
        "legend.borderpad": 0.4,
        "legend.labelspacing": 0.3,
        "legend.handlelength": 1.6,
        "legend.handletextpad": 0.5,
        # figura
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


_MATH_ESCAPE = {"-": r"\text{-}", " ": r"\ "}


def mathtt(text: str) -> str:
    """Nome di algoritmo in monospace mathtext, come nella figura di riferimento.

    'wPPO-BH' -> '$\\mathtt{\\omega PPO\\text{-}BH}$' (la w iniziale diventa omega).
    """
    out = []
    # \omega deve stare fuori da \mathtt per avere la lettera greca corsiva
    for chunk in re.split(r"(ω)", text):
        if chunk == "ω":
            out.append(r"\omega")
        elif chunk:
            for ch, esc in _MATH_ESCAPE.items():
                chunk = chunk.replace(ch, esc)
            out.append(r"\mathtt{%s}" % chunk)
    return "$" + "".join(out) + "$"


def finalize_axes(ax, xmax=None, xlabel=True, ylabel=True,
                  xlabel_text=None, ylabel_text=None) -> None:
    """Etichette e limiti coerenti con lo stile di riferimento."""
    xlabel_text = R.get("figure", "xlabel") if xlabel_text is None else xlabel_text
    ylabel_text = R.get("figure", "ylabel") if ylabel_text is None else ylabel_text
    if xlabel:
        ax.set_xlabel(xlabel_text)
    if ylabel:
        ax.set_ylabel(ylabel_text)
    if xmax is not None:
        ax.set_xlim(0, xmax)
    # tick "tondi" sull'asse x (0.0, 0.2, ... come nella figura di riferimento)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
    ax.tick_params(top=False, right=False)


def save(fig, outdir, name: str, formats=("png", "pdf")) -> list[str]:
    """Salva la figura nei formati richiesti; restituisce i path scritti."""
    from pathlib import Path

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        path = outdir / f"{name}.{fmt}"
        fig.savefig(path, format=fmt)
        written.append(str(path))
    plt.close(fig)
    return written
