#!/usr/bin/env python
"""Prestazione finale (media dell'ultima parte della curva di eval) a barre.

Utile per i confronti sintetici fra molte configurazioni: una barra per serie,
raggruppate lungo l'asse x da `--x`, con errore standard sui seed.

Condivide con plot_curves.py la selezione (filtri, --runs-file, --state), la
scelta automatica delle serie e la nomenclatura: cambia solo il disegno.

Esempi:
    python plots/scripts/plot_final.py \
        --filter family=RT-PPO env=Hopper-v5 setting=2 \
        --x window --hue is_type opc --baseline family=PPO env=Hopper-v5 \
        --name hopper_s2_final

    python plots/scripts/plot_final.py --filter family=RT-PPO setting=1 window=4 \
        --panels env --x is_type --hue opc --name final_by_env
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np

import _bootstrap  # noqa: F401
from _common import (add_aggregation_args, add_output_args, add_selection_args,
                     spec_from_args)
from rtplots import figure as F
from rtplots import labels as L
from rtplots import style as S
from rtplots.curves import final_performance, load_curves
from rtplots.index import load_index
from rtplots.select import select_runs


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_selection_args(p)
    add_aggregation_args(p)
    add_output_args(p, default_name="final")
    g = p.add_argument_group("barre")
    g.add_argument("--x", default="window", help="colonna sull'asse x")
    g.add_argument("--panels", default=None, help="colonna che genera i sotto-pannelli")
    g.add_argument("--last-frac", type=float, default=0.1,
                   help="frazione finale della curva su cui mediare (default 10%%)")
    g.add_argument("--err", default="se", choices=["se", "std", "none"])
    g.add_argument("--ylabel", default="Final Mean Return")
    g.add_argument("--ylim", nargs=2, type=float, default=None)
    g.add_argument("--panel-size", nargs=2, type=float, default=[3.4, 2.6])
    g.add_argument("--legend-loc", default="best")
    g.add_argument("--font-scale", type=float, default=1.0)
    g.add_argument("--paper-names", dest="paper", action="store_true", default=True)
    g.add_argument("--raw-names", dest="paper", action="store_false")
    return p.parse_args()


def main():
    args = parse_args()
    S.apply_style(args.font_scale)

    spec = spec_from_args(args)
    index = load_index()
    sel = F.select(index, spec)
    if sel.empty:
        raise SystemExit("Nessun run corrisponde ai filtri.")
    print(f"[plot] {len(sel)} run selezionati")

    hue = list(spec.hue) if spec.hue else F.auto_hue(sel, exclude=(args.x, args.panels))
    print(f"[plot] serie per: {', '.join(hue)}")
    merged = F.merged_dims(sel, hue, (args.x, args.panels))
    if merged:
        print(f"[plot] ATTENZIONE: {', '.join(merged)} variano ma non separano le barre")
    group_cols = sorted(set(hue) | {"family", args.x} |
                        ({args.panels} if args.panels else set()))
    curves = load_curves(sel, source=spec.source, metric=spec.metric)
    if curves.empty:
        raise SystemExit("Nessuna curva disponibile per la selezione.")
    agg, per_run = final_performance(curves, sel, group_cols, last_frac=args.last_frac)

    fields = tuple(spec.label_fields or hue) + ("family",)
    agg["label"] = [L.series_label(r, fields=fields, paper=spec.paper)
                    for r in agg.to_dict("records")]
    order = agg.drop_duplicates("label").sort_values(hue)["label"].tolist()
    colors = dict(zip(order, S.color_cycle(len(order))))

    base_stat = None
    if spec.baseline.filters:
        # la baseline si cerca sempre nell'indice completo, non nella selezione
        base_sel = select_runs(index, spec.baseline.filters, state=spec.state)
        if not base_sel.empty:
            base_curves = load_curves(base_sel, source=spec.source, metric=spec.metric)
            base_agg, _ = final_performance(base_curves, base_sel, ["family"],
                                            last_frac=args.last_frac)
            base_stat = base_agg.iloc[0]
            print(f"[plot] baseline {base_stat['family']}: {base_stat['mean']:.1f}")

    panel_vals = sorted(agg[args.panels].dropna().unique()) if args.panels else [None]
    fig, axes = plt.subplots(1, len(panel_vals), squeeze=False, sharey=True,
                             figsize=(args.panel_size[0] * len(panel_vals), args.panel_size[1]))

    for k, pv in enumerate(panel_vals):
        ax = axes[0][k]
        sub = agg[agg[args.panels] == pv] if args.panels else agg
        xvals = sorted(sub[args.x].dropna().unique())
        n_series = len(order)
        width = 0.8 / max(n_series, 1)
        for si, lab in enumerate(order):
            g = sub[sub.label == lab].set_index(args.x).reindex(xvals)
            pos = np.arange(len(xvals)) + (si - (n_series - 1) / 2) * width
            err = None if args.err == "none" else g[args.err].to_numpy()
            ax.bar(pos, g["mean"].to_numpy(), width=width * 0.92, label=lab,
                   color=colors[lab], edgecolor="none", yerr=err,
                   error_kw=dict(elinewidth=0.9, capsize=2, ecolor="0.25"))
        if base_stat is not None:
            ax.axhline(base_stat["mean"], color=S.BASELINE_COLOR, lw=S.BASELINE_WIDTH,
                       ls="--", label=S.mathtt(str(base_stat["family"])), zorder=1)
            if args.err != "none" and np.isfinite(base_stat.get(args.err, np.nan)):
                ax.axhspan(base_stat["mean"] - base_stat[args.err],
                           base_stat["mean"] + base_stat[args.err],
                           color=S.BASELINE_COLOR, alpha=S.BAND_ALPHA, lw=0, zorder=0)
        ax.set_xticks(np.arange(len(xvals)))
        ax.set_xticklabels([L.panel_title(args.x, v, spec.paper) for v in xvals])
        ax.set_xlabel({"window": r"$\omega$"}.get(args.x, args.x))
        if k == 0:
            ax.set_ylabel(args.ylabel)
        if args.panels:
            ax.set_title(L.panel_title(args.panels, pv, spec.paper))
        if args.ylim:
            ax.set_ylim(*args.ylim)
        ax.tick_params(top=False, right=False)
        if k == 0:
            ax.legend(loc=args.legend_loc)

    fig.tight_layout()
    for p in S.save(fig, args.outdir, args.name, formats=args.formats):
        print(f"[plot] scritto {p}")
    if args.dump_csv:
        agg.to_csv(f"{args.outdir}/{args.name}.csv", index=False)
        per_run.to_csv(f"{args.outdir}/{args.name}_per_run.csv", index=False)
        print(f"[plot] scritti i csv in {args.outdir}")

    print(agg[[*group_cols, "mean", "std", "n"]].to_string(index=False))


if __name__ == "__main__":
    main()
