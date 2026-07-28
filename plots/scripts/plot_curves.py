#!/usr/bin/env python
"""Curve (metrica vs environment steps) su griglia di pannelli.

E' lo script generico: si sceglie cosa filtrare, cosa plottare (--metric), cosa
mettere sulle righe, sulle colonne e cosa distinguere con il colore.

Le metriche di eval si leggono dai file locali, tutte le altre dalla history W&B
(scaricate una volta e poi in cache). L'elenco completo:
    python plots/scripts/plot_curves.py --list-metrics

Con --runs-file la figura e' esattamente quella dell'anteprima del selettore:
righe, colonne, colori, banda e baseline arrivano dalla selezione salvata, e la
riga di comando serve solo a scavalcare quel che si vuole cambiare.

Esempi:
    # Una riga per setting, una colonna per omega, colori = IS x critic (figura del paper)
    python plots/scripts/plot_curves.py \
        --filter family=RT-PPO env=Hopper-v5 \
        --rows setting --cols window --hue is_type opc \
        --baseline family=PPO env=Hopper-v5 \
        --row-captions auto --name hopper_rtppo

    # GePPO vs RT-PPO sui quattro environment coperti in W&B
    python plots/scripts/plot_curves.py \
        --filter family=GePPO,RT-PPO setting=1 is_type=N \
        --cols env --hue family window --share col --name geppo_vs_rtppo

    # Una diagnostica invece del return
    python plots/scripts/plot_curves.py --metric diagnostics_ess/final_naive_ess_mean \
        --filter family=RT-PPO env=Hopper-v5 setting=1 --cols window --name hopper_ess
"""
import argparse

import _bootstrap  # noqa: F401
from _common import (add_aggregation_args, add_grid_args, add_output_args,
                     add_selection_args, report, spec_from_args)
from rtplots import figure as F
from rtplots import style as S
from rtplots.index import load_index
from rtplots.metrics import DEFAULT_METRIC, METRIC_GROUPS


def print_metrics():
    for group, items in METRIC_GROUPS:
        print(f"\n{group}")
        for key, label, _, source, _ in items:
            print(f"  {key:<45} {label} [{source}]")
    print("\nQualsiasi altra chiave loggata su W&B e' accettata.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metric", default=None,
                   help=f"cosa plottare (default {DEFAULT_METRIC}); --list-metrics per l'elenco")
    p.add_argument("--list-metrics", action="store_true", help="stampa le metriche note ed esce")
    add_selection_args(p)
    add_aggregation_args(p)
    add_grid_args(p)
    add_output_args(p, default_name="curves")
    p.add_argument("--ylabel", default=None, help="etichetta asse y (default: dalla metrica)")
    args = p.parse_args()
    if args.list_metrics:
        return print_metrics()

    S.apply_style(args.font_scale)
    spec = spec_from_args(args)
    try:
        fig, series = F.build(load_index(), spec)
    except ValueError as exc:
        raise SystemExit(str(exc))
    report(series, args, S.save(fig, args.outdir, args.name, formats=args.formats))


if __name__ == "__main__":
    main()
