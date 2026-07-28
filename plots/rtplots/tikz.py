"""Export della figura in pgfplots (`.tex`) tramite tikzplotlib, un file per pannello.

Il `.tex` prodotto e' autonomo: `\\begin{tikzpicture}` con i `\\definecolor`, le
bande come `\\path[fill=..., opacity=0.2]` e una `\\addplot` per serie — cioe' i
dati veri dentro il sorgente, non un PDF incluso. Si compila con `pgfplots` e si
ricolora/riscala da LaTeX senza rigenerare niente.

**Un pannello, un file.** Una griglia m×n non diventa un `.tex` con m×n assi: le
figure di un paper si compongono in LaTeX (subfigure), non in matplotlib. Chi
esporta i sorgenti vuole i singoli riquadri.

tikzplotlib e' fermo alla 0.10.1 (2022) e non regge matplotlib 3.6+ ne' numpy 2:
`_compat()` rimette i nomi che si aspetta. Sono alias, non rattoppi alla logica —
se un giorno la libreria verra' sostituita, qui si toglie una funzione sola.
"""
from __future__ import annotations

import re

# amsmath serve per \text{} dentro le voci di legenda (ωPPO-BH e simili)
PREAMBLE = (r"% Nel preambolo del documento: \usepackage{pgfplots,amsmath}"
            r" \pgfplotsset{compat=1.18}")


def _compat() -> None:
    """Alias dei nomi rimossi da matplotlib/numpy che tikzplotlib usa ancora."""
    import numpy as np
    import matplotlib.backends.backend_pgf as pgf
    from matplotlib.legend import Legend

    if not hasattr(pgf, "common_texification"):        # rinominata in mpl 3.6
        pgf.common_texification = pgf._tex_escape
    if not hasattr(Legend, "legendHandles"):           # rinominata in mpl 3.9
        Legend.legendHandles = property(lambda self: self.legend_handles)
    if not hasattr(Legend, "_ncol"):                   # rinominata in mpl 3.6
        Legend._ncol = property(lambda self: self._ncols)
    for old, new in (("float_", "float64"), ("alltrue", "all"), ("bool8", "bool_")):
        if not hasattr(np, old):                       # rimossi in numpy 2.0
            setattr(np, old, getattr(np, new))


def unavailable_reason() -> str | None:
    """None se l'export .tex e' possibile, altrimenti perche' non lo e'."""
    try:
        _compat()
        import tikzplotlib  # noqa: F401
    except Exception as exc:
        return (f"Export LaTeX non disponibile ({type(exc).__name__}: {exc}). "
                f"Serve tikzplotlib: pip install -r requirements.txt")
    return None


TOKEN = "RTPLOTSLABEL%d"


def _use_latex_names(fig, styles: dict) -> dict:
    """Sostituisce le etichette con segnaposto, e dice a cosa vanno rimessi.

    I nomi `latex` di style.toml sono macro (`\\uppo`): matplotlib non le sa
    disegnare e tikzplotlib le escaperebbe. Si passa quindi da un segnaposto
    innocuo, rimpiazzato nel codice generato — cosi' il `.tex` esce con le macro
    del paper mentre l'anteprima continua a mostrare i nomi leggibili.
    """
    mapping = {}
    for ax in fig.axes:
        for i, line in enumerate(ax.get_lines()):
            latex = (styles.get(line.get_label()) or {}).get("latex")
            if not latex:
                continue
            token = TOKEN % len(mapping)
            mapping[token] = str(latex)
            line.set_label(token)
        legend = ax.get_legend()
        if legend is not None and mapping:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, loc=legend._loc, ncol=legend._ncols)
    return mapping


def _add_axis_options(code: str, options) -> str:
    """Aggiunge righe dentro `\\begin{axis}[...]`, in coda: le ultime vincono."""
    extra = [str(o).rstrip(",") for o in options if str(o).strip()]
    if not extra:
        return code
    # il blocco di opzioni e' quello che sta fra "\begin{axis}[" e la prima "]"
    # a inizio riga: pgfplots non annida parentesi quadre a quel livello
    head, sep, rest = code.partition("\\begin{axis}[\n")
    if not sep:
        return code
    body, closer, tail = rest.partition("\n]\n")
    if not closer:
        return code
    return head + sep + body + ",\n" + ",\n".join(extra) + closer + tail


def figure_to_tex(fig, header: str = "", styles: dict | None = None) -> str:
    """Codice tikzpicture della figura (che deve avere un solo asse).

    `styles` e' la mappa etichetta -> stile prodotta da `figure.prepare`: da li'
    arrivano i nomi LaTeX delle serie.
    """
    _compat()
    import tikzplotlib

    from . import rules as R

    names = _use_latex_names(fig, styles or {})
    code = tikzplotlib.get_tikz_code(figure=fig)
    for token, latex in names.items():
        code = code.replace(token, latex)
    code = _add_axis_options(code, R.get("latex", "axis_options") or [])
    # tikzplotlib scrive "\path [draw=...]": lo spazio dopo il comando e' legale
    # ma sporca i diff quando si rigenera la stessa figura
    code = re.sub(r"\\(path|addplot) \[", r"\\\1[", code)
    bits = [b for b in (header, R.latex_macros_comment()) if b]
    return "\n".join(bits + [code]) if bits else code
