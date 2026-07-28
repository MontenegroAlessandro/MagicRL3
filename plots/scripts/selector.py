#!/usr/bin/env python
"""Selettore interattivo delle run: filtri a schermo, copertura e anteprima.

    .venv/bin/python plots/scripts/selector.py            # http://127.0.0.1:8770
    .venv/bin/python plots/scripts/selector.py --port 9000

Serve una pagina locale (nessuna dipendenza esterna: solo libreria standard piu'
quelle gia' usate dai plot). L'indice viene riletto a ogni query, quindi dopo un
build_index.py i nuovi run compaiono senza riavviare.

Il bottone "Salva selezione" scrive /storage/fis1/plots_cache/selection.json con
dentro la figura completa (filtri, griglia, colori, baseline): da quel momento
`plot_curves.py --runs-file` rifa' esattamente quella figura.

Il codice sta in rtplots/webui/: `api.py` la logica, `server.py` il trasporto.
"""
import argparse

import _bootstrap  # noqa: F401
from rtplots.webui import serve


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8770)
    p.add_argument("--host", default="127.0.0.1",
                   help="127.0.0.1 (default, usa il port forwarding di VSCode) o 0.0.0.0")
    args = p.parse_args()
    serve(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
