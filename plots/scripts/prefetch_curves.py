#!/usr/bin/env python
"""Scarica in cache le curve delle run che non hanno gli evaluations.npz locali.

Le run del paper vivono solo su W&B: senza cache ogni figura che le include
paga una richiesta di rete per run. Questo script riempie in un colpo solo
/storage/fis1/plots_cache/curves/ (~9 KB per run per metrica).

    python plots/scripts/prefetch_curves.py                    # eval, tutte le run del paper
    python plots/scripts/prefetch_curves.py --filter env=Hopper-v5
    python plots/scripts/prefetch_curves.py --metric train/loss
"""
import argparse
from concurrent.futures import ThreadPoolExecutor

import _bootstrap  # noqa: F401
from rtplots.curves import build_eval_map, curve_from_wandb
from rtplots.index import load_index
from rtplots.metrics import DEFAULT_METRIC
from rtplots.sources import DEFAULT_PROJECT
from rtplots.select import select_runs


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--filter", nargs="*", default=[], help="filtri come negli altri script")
    p.add_argument("--metric", default=DEFAULT_METRIC)
    p.add_argument("--state", default="finished")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--all", action="store_true",
                   help="anche le run che hanno gia' un .npz locale")
    args = p.parse_args()

    sel = select_runs(load_index(), args.filter, state=args.state)
    if not args.all:
        eval_map = build_eval_map()
        sel = sel[~sel.run_id.isin(eval_map)]
    todo = list(zip(sel.run_id, sel.project.fillna(DEFAULT_PROJECT)))
    print(f"[prefetch] {len(todo)} run, metrica {args.metric}")

    ok = err = 0
    def fetch(item):
        run_id, project = item
        try:
            return len(curve_from_wandb(run_id, project=project, metric=args.metric))
        except Exception as exc:
            print(f"[prefetch] {project}/{run_id}: {exc}")
            return None

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, n in enumerate(pool.map(fetch, todo), 1):
            ok += n is not None
            err += n is None
            if i % 100 == 0:
                print(f"[prefetch]   {i}/{len(todo)}")
    print(f"[prefetch] in cache: {ok}, falliti: {err}")


if __name__ == "__main__":
    main()
