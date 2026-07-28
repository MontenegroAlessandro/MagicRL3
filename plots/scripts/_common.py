"""Argomenti CLI condivisi, e come diventano uno `FigureSpec`.

Gli script non preparano piu' i dati: costruiscono lo spec e lo passano a
`rtplots.figure`, la stessa pipeline che usa il selettore. Cosi' «rifai da CLI
la figura che vedo nella pagina» e' garantito, non sperato.

Le opzioni che descrivono *la figura* (righe, colonne, colori, banda, metrica...)
hanno default `None`: quando si parte da una selezione salvata vince quello che
c'e' dentro la selezione, e la riga di comando serve solo a scavalcarlo.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
from rtplots import selection as SEL
from rtplots.figure import BaselineSpec, FigureSpec
from rtplots.metrics import DEFAULT_METRIC
from rtplots.paths import OUTPUT_DIR, SELECTION_JSON


def add_selection_args(p):
    g = p.add_argument_group("selezione")
    g.add_argument("--filter", nargs="*", default=[],
                   help="filtri sull'indice, es. env=Hopper-v5 setting=2 window=2,4")
    g.add_argument("--state", default=None, help="finished (default) | any | crashed")
    g.add_argument("--runs-file", nargs="?", const=str(SELECTION_JSON), default=None,
                   metavar="PATH",
                   help="usa la selezione salvata dal selettore interattivo "
                        f"(senza argomento: {SELECTION_JSON})")
    g.add_argument("--baseline", nargs="*", default=[],
                   help="filtri per la baseline (disegnata in nero in ogni pannello)")
    g.add_argument("--baseline-epochs", default=None,
                   help="seconda baseline tratteggiata a epoche moltiplicate: "
                        "un numero (2|4|8) o 'follow_window' (segue l'ω del pannello)")
    g.add_argument("--hue", nargs="*", default=None,
                   help="colonne che definiscono le serie colorate; per default sono "
                        "automatiche (tutte le dimensioni che variano nella selezione, "
                        "escluse quelle su righe/colonne)")
    g.add_argument("--hue-order", nargs="*", default=None)
    g.add_argument("--label-fields", nargs="*", default=None,
                   help="campi mostrati in legenda (default: --hue). Es. 'is_type opc window' "
                        "per avere anche omega, come nella figura del paper")
    g.add_argument("--min-seeds", type=int, default=1)
    return p


def add_aggregation_args(p):
    g = p.add_argument_group("aggregazione")
    g.add_argument("--band", default=None,
                   choices=["se", "std", "ci95", "iqr", "minmax", "none"],
                   help="banda: errore standard (default), std, IC 95%%, IQR, min-max")
    g.add_argument("--smooth", type=int, default=None,
                   help="finestra della media mobile (default 5)")
    g.add_argument("--grid-points", type=int, default=None)
    g.add_argument("--source", default="auto", choices=["auto", "local", "wandb"])
    return p


def add_grid_args(p):
    g = p.add_argument_group("griglia e aspetto")
    g.add_argument("--rows", default=None, help="colonna dell'indice per le righe")
    g.add_argument("--cols", default=None, help="colonna dell'indice per le colonne")
    g.add_argument("--xmax", type=float, default=None, help="in step, es. 1e6")
    g.add_argument("--xscale", type=float, default=1e6)
    g.add_argument("--ylim", nargs=2, type=float, default=None)
    g.add_argument("--logy", action="store_true")
    g.add_argument("--share", default=None, choices=["none", "row", "col", "all"])
    g.add_argument("--panel-size", nargs=2, type=float, default=None)
    g.add_argument("--legend", default=None, choices=["panel", "figure", "first", "none"])
    g.add_argument("--legend-loc", default=None)
    g.add_argument("--legend-ncol", type=int, default=None)
    g.add_argument("--titles", default=None, choices=["auto", "off"])
    g.add_argument("--label-mode", default=None, choices=["all", "edge"])
    g.add_argument("--sublabels", action="store_true", help="(a) (b) (c) sotto i pannelli")
    g.add_argument("--row-captions", default=None,
                   help="off | auto | testi separati da ';'")
    g.add_argument("--suptitle", default=None)
    g.add_argument("--font-scale", type=float, default=1.0)
    g.add_argument("--paper-names", dest="paper", action="store_true", default=True)
    g.add_argument("--raw-names", dest="paper", action="store_false",
                   help="nomi del codice (RT-PPO IS=N) invece di quelli del paper")
    return p


def add_output_args(p, default_name="figure"):
    g = p.add_argument_group("output")
    g.add_argument("--name", default=default_name)
    g.add_argument("--outdir", default=str(OUTPUT_DIR))
    g.add_argument("--formats", nargs="*", default=["png", "pdf"])
    g.add_argument("--dump-csv", action="store_true", help="salva anche i dati aggregati")
    return p


def spec_from_args(args, metric: str | None = None) -> FigureSpec:
    """Argomenti (piu' l'eventuale selezione salvata) -> FigureSpec.

    Ordine di precedenza: riga di comando > selezione salvata > default.
    """
    spec = FigureSpec()
    runs_file = getattr(args, "runs_file", None)
    if runs_file:
        spec, data = SEL.spec_from(runs_file)
        print(f"[plot] selezione del {data.get('saved_at')}: {data.get('n_runs')} run "
              f"({' '.join(data.get('filter_args') or []) or 'nessun filtro'})")
    else:
        spec.state = "finished"

    def pick(attr, dest=None, cast=None):
        value = getattr(args, attr, None)
        if value is None:
            return
        setattr(spec, dest or attr, cast(value) if cast else value)

    # cosa
    spec.filters = list(getattr(args, "filter", []) or [])
    pick("state")
    if metric or getattr(args, "metric", None):
        spec.metric = metric or args.metric
    spec.metric = spec.metric or DEFAULT_METRIC
    pick("source")
    # serie
    for attr in ("rows", "cols", "hue", "hue_order", "label_fields", "min_seeds"):
        pick(attr)
    # aggregazione
    for attr in ("band", "smooth", "grid_points", "xmax"):
        pick(attr)
    # aspetto
    for attr in ("paper", "share", "legend", "legend_loc", "legend_ncol", "titles",
                 "label_mode", "sublabels", "row_captions", "suptitle", "xscale", "logy"):
        pick(attr)
    pick("ylim", cast=tuple)
    pick("panel_size", cast=tuple)
    pick("ylabel")
    # baseline: i filtri della CLI hanno la precedenza su quella della selezione
    if getattr(args, "baseline", None):
        spec.baseline = BaselineSpec(filters=list(args.baseline))
    if getattr(args, "baseline_epochs", None):
        spec.baseline.epochs = args.baseline_epochs
    return spec


def report(series, args, paths):
    for p in paths:
        print(f"[plot] scritto {p}")
    if args.dump_csv:
        csv = f"{args.outdir}/{args.name}.csv"
        series.agg.to_csv(csv, index=False)
        print(f"[plot] scritto {csv}")
    print("[plot] seed per serie:\n"
          + series.agg.groupby("label")["n_seeds"].max().to_string())
