"""Disegno della griglia di pannelli (condiviso da plot_curves.py e dal selettore).

Riceve un DataFrame gia' aggregato con colonne [step, mean, lo, hi, n_seeds, label]
piu' le colonne usate per righe/colonne, e produce la figura nello stile di
riferimento: pannelli con box, banda ombreggiata, legenda interna, palette IBM.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt

from . import labels as L
from . import rules as R
from . import style as S

LETTERS = "abcdefghijklmnopqrstuvwxyz"


@dataclass
class GridOptions:
    rows: str | None = None
    cols: str | None = None
    xscale: float = 1e6
    xmax: float | None = None
    ylim: tuple | None = None
    share: str = "row"
    panel_size: tuple = (3.1, 2.3)
    legend: str = "panel"
    legend_loc: str = "best"
    legend_ncol: int = 1
    titles: str = "auto"
    label_mode: str = "all"
    sublabels: bool = False
    row_captions: str = "off"
    suptitle: str | None = None
    xlabel: str = field(default_factory=lambda: R.get("figure", "xlabel"))
    ylabel: str = field(default_factory=lambda: R.get("figure", "ylabel"))
    logy: bool = False
    paper: bool = True
    hue: list = field(default_factory=list)


def panel_values(df, col):
    if col is None:
        return [None]
    return sorted(df[col].dropna().unique().tolist())


def draw_grid(agg, order, styles, opts: GridOptions, base_agg=None):
    """`styles`: etichetta -> {color, width, style, band_alpha} (vedi figure.py)."""
    row_vals = panel_values(agg, opts.rows)
    col_vals = panel_values(agg, opts.cols)
    nrow, ncol = len(row_vals), len(col_vals)
    sharey = {"none": False, "row": "row", "col": "col", "all": True}[opts.share]
    fig, axes = plt.subplots(nrow, ncol, sharex=True, sharey=sharey, squeeze=False,
                             figsize=(opts.panel_size[0] * ncol, opts.panel_size[1] * nrow))

    for i, rv in enumerate(row_vals):
        for j, cv in enumerate(col_vals):
            ax = axes[i][j]
            sub = agg
            if opts.rows:
                sub = sub[sub[opts.rows] == rv]
            if opts.cols:
                sub = sub[sub[opts.cols] == cv]

            # baseline sotto le altre curve, sempre nera
            if base_agg is not None:
                b = base_agg
                for fieldname, value in ((opts.rows, rv), (opts.cols, cv)):
                    if fieldname and fieldname in b.columns and b[fieldname].notna().any():
                        b = b[b[fieldname] == value]
                for lab, g in b.groupby("label"):
                    g = g.sort_values("step")
                    x = g["step"] / opts.xscale
                    # colonna 'dash': permette due baseline nello stesso pannello
                    # (p.es. PPO base continua + PPO a epoche moltiplicate)
                    dashed = bool(g["dash"].iloc[0]) if "dash" in g.columns else False
                    ax.plot(x, g["mean"], color=S.baseline_color(),
                            lw=S.baseline_width(),
                            ls="--" if dashed else "-", label=lab, zorder=1)
                    ax.fill_between(x, g["lo"], g["hi"], color=S.baseline_color(),
                                    alpha=S.band_alpha(), lw=0, zorder=0)

            for lab in order:
                g = sub[sub.label == lab].sort_values("step")
                if g.empty:
                    continue
                x = g["step"] / opts.xscale
                # colore, spessore e tratto vengono dalle regole [[series]] di
                # style.toml; il fallback e' la palette + i valori di [lines]
                st = styles.get(lab, {})
                ax.plot(x, g["mean"], color=st["color"], lw=st["width"],
                        ls=st["style"], label=lab, zorder=3)
                ax.fill_between(x, g["lo"], g["hi"], color=st["color"],
                                alpha=st["band_alpha"], lw=0, zorder=2)

            if opts.titles == "auto" and i == 0 and opts.cols:
                title = L.panel_title(opts.cols, cv, opts.paper)
                if title:
                    ax.set_title(title)

            show_x = opts.label_mode == "all" or i == nrow - 1
            show_y = opts.label_mode == "all" or j == 0
            S.finalize_axes(ax, xmax=(opts.xmax / opts.xscale) if opts.xmax else None,
                            xlabel=show_x, ylabel=show_y,
                            xlabel_text=opts.xlabel, ylabel_text=opts.ylabel)
            if opts.label_mode == "all":
                # con sharex/sharey matplotlib nasconde i tick interni: nello stile
                # di riferimento ogni pannello ha i propri tick
                ax.tick_params(labelbottom=True, labelleft=True)
            if opts.logy:
                ax.set_yscale("log")
            if opts.ylim:
                ax.set_ylim(*opts.ylim)
            if opts.legend == "panel" or (opts.legend == "first" and i == 0 and j == 0):
                handles, lbls = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(handles, lbls, loc=opts.legend_loc, ncol=opts.legend_ncol)
            if opts.sublabels:
                ax.text(0.5, -0.34, f"({LETTERS[i * ncol + j]})", transform=ax.transAxes,
                        ha="center", va="top")

    if opts.legend == "figure":
        handles, lbls = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, lbls, loc="lower center", ncol=min(len(lbls), 4),
                   bbox_to_anchor=(0.5, -0.03))
    if opts.suptitle:
        fig.suptitle(opts.suptitle)

    fig.tight_layout()

    if opts.row_captions != "off":
        if opts.row_captions == "auto" and opts.rows == "setting":
            caps = [f"({LETTERS[i]}) {L.SETTING_CAPTIONS.get(int(rv), str(rv))}"
                    for i, rv in enumerate(row_vals)]
        elif opts.row_captions == "auto":
            caps = [f"({LETTERS[i]}) {L.panel_title(opts.rows, rv, opts.paper)}"
                    for i, rv in enumerate(row_vals)]
        else:
            caps = opts.row_captions.split(";")
        # spazio extra fra le righe per la caption, poi posizionamento sotto le
        # etichette dell'asse x (tight bbox della riga, in coordinate figura)
        fig.subplots_adjust(hspace=0.62)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        inv = fig.transFigure.inverted()
        for i, cap in enumerate(caps[:nrow]):
            y = min(axes[i][j].get_tightbbox(renderer).transformed(inv).y0
                    for j in range(ncol))
            fig.text(0.5, y - 0.012, cap, ha="center", va="top", style="italic")

    return fig
