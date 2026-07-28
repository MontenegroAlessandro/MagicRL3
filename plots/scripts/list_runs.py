#!/usr/bin/env python
"""Interroga l'indice: quante run/seed esistono per ogni combinazione.

Esempi:
    python plots/scripts/list_runs.py --by family env window setting
    python plots/scripts/list_runs.py --filter family=RT-PPO env=Hopper-v5 \
        --by setting window is_type
    python plots/scripts/list_runs.py --filter family=GePPO --state any --by state env window
"""
import argparse

import pandas as pd

import _bootstrap  # noqa: F401
from rtplots.curves import build_eval_map
from rtplots.index import load_index
from rtplots.select import coverage, select_runs


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--filter", nargs="*", default=[], help="es. env=Hopper-v5 window=4")
    p.add_argument("--by", nargs="*", default=["family", "env", "window", "setting"])
    p.add_argument("--state", default="finished", help="finished | any | crashed,running")
    p.add_argument("--check-curves", action="store_true",
                   help="verifica quante curve locali esistono per la selezione")
    p.add_argument("--full", action="store_true", help="stampa una riga per run")
    args = p.parse_args()

    df = load_index()
    sel = select_runs(df, args.filter, state=args.state)
    print(f"[list] {len(sel)} run selezionati su {len(df)}")

    if args.full:
        cols = [c for c in ["run_id", "family", "env", "window", "setting", "is_type",
                            "opc", "fresh_adv", "adaptive_lr", "sampling", "seed",
                            "state", "campaign"] if c in sel.columns]
        with pd.option_context("display.max_rows", None, "display.width", 250):
            print(sel[cols].sort_values(cols[1:5]).to_string(index=False))
        return

    if args.check_curves:
        emap = build_eval_map()
        sel["has_local_curve"] = sel.run_id.isin(emap)
        args.by = list(args.by) + ["has_local_curve"]

    cov = coverage(sel, args.by)
    with pd.option_context("display.max_rows", None, "display.width", 250):
        print(cov.to_string(index=False))


if __name__ == "__main__":
    main()
