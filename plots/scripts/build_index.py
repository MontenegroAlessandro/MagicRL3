#!/usr/bin/env python
"""Costruisce/aggiorna la cache dei metadati dei run W&B e la mappa delle curve locali.

Indicizza tutti i progetti registrati in rtplots/sources/: le campagne nuove e
i progetti del paper (uno per environment), ognuno con le sue convenzioni.

    python plots/scripts/build_index.py            # aggiornamento incrementale
    python plots/scripts/build_index.py --force    # riscarica tutte le config
    python plots/scripts/build_index.py --projects rebuttal   # solo un progetto
"""
import argparse

import _bootstrap  # noqa: F401
from rtplots.curves import build_eval_map
from rtplots.index import build_index
from rtplots.sources import ALL_PROJECTS


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--force", action="store_true", help="ignora la cache e riscarica tutto")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--projects", nargs="+", default=ALL_PROJECTS,
                   help=f"progetti W&B da indicizzare (default: {' '.join(ALL_PROJECTS)})")
    p.add_argument("--skip-eval-map", action="store_true",
                   help="non reindicizzare gli evaluations.npz in storage")
    args = p.parse_args()

    df = build_index(force=args.force, workers=args.workers, projects=args.projects)
    if not args.skip_eval_map:
        build_eval_map(refresh=True, verbose=True)

    print("\n[index] run per fonte:")
    print(df.source.value_counts(dropna=False).to_string())
    print("\n[index] run per famiglia:")
    print(df.family.value_counts(dropna=False).to_string())
    print("\n[index] run per stato:")
    print(df.state.value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
