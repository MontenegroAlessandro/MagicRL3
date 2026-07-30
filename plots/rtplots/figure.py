"""Dalla selezione alla figura: una sola pipeline, usata da CLI e selettore.

Prima esistevano due copie di questo codice — `scripts/_common.prepare_series`
per la riga di comando e `scripts/selector.render_figure` per la pagina — e si
erano gia' allontanate: ordine dei colori diverso sui campi booleani, opzioni di
griglia diverse, due implementazioni della baseline di cui solo quella del sito
sapeva seguire ω. Risultato: l'anteprima e la figura ricostruita da CLI sulla
stessa selezione non erano la stessa figura.

Qui la richiesta e' un oggetto solo, `FigureSpec`, serializzabile: e' anche il
formato con cui il selettore salva le selezioni, quindi «rifammi questa figura»
e' letteralmente rileggere lo spec.
"""
from __future__ import annotations

from dataclasses import (asdict, dataclass, field, fields as dataclass_fields,
                         replace as dataclass_replace)

import pandas as pd

from . import labels as L
from . import rules as R
from . import schema
from . import style as S
from .curves import aggregate, load_curves
from .grid import GridOptions, draw_grid
from .metrics import DEFAULT_METRIC, metric_info
from .select import select_runs

SPEC_VERSION = 2


@dataclass
class BaselineSpec:
    """Curve nere di riferimento, cercate sempre nell'indice completo.

    Due modi di indicarle, stesso disegno:
      - `filters`: sintassi degli script (`family=PPO,GePPO-original env=Hopper-v5`);
      - `family`: le pillole del selettore.

    `family` accetta una famiglia sola (forma vecchia, ancora nelle selezioni
    salvate) o un elenco: le baseline scelte si disegnano tutte, nere, una per
    tratteggio (vedi `[lines].baseline_styles` in style.toml).

    `epochs` aggiunge una seconda baseline tratteggiata a epoche moltiplicate:
    un numero (×2, ×4, ×8) oppure "follow_window", che in ogni pannello usa il
    PPO con lo stesso moltiplicatore dell'ω del pannello — il confronto «stesso
    numero di update» delle figure del paper.
    """

    filters: list = field(default_factory=list)
    family: str | list | None = None
    epochs: str = ""

    def families(self) -> list:
        """Le famiglie scelte, sempre come lista (vuota se non ce n'e')."""
        fam = self.family
        if not fam:
            return []
        return [f for f in ([fam] if isinstance(fam, str) else list(fam)) if f]

    def active(self) -> bool:
        return bool(self.filters or self.families())


@dataclass
class FigureSpec:
    """Tutto quello che serve per disegnare una figura, e nient'altro."""

    # cosa
    run_ids: list | None = None          # selezione esplicita (dal selettore)
    filters: list = field(default_factory=list)
    state: str | None = "finished"
    metric: str = DEFAULT_METRIC
    source: str = "auto"                 # auto | local | wandb
    # come si dividono le curve
    rows: str | None = None
    cols: str | None = None
    hue: list | None = None              # None = automatico
    hue_order: list | None = None
    label_fields: list | None = None     # None = come hue
    min_seeds: int = 1
    # aggregazione (i default vengono da plots/style.toml, vedi rules.py)
    band: str = field(default_factory=lambda: R.get("lines", "band"))
    smooth: int = field(default_factory=lambda: int(R.get("lines", "smooth")))
    grid_points: int | None = None
    xmax: float | None = None
    baseline: BaselineSpec = field(default_factory=BaselineSpec)
    # aspetto
    paper: bool = True
    share: str = field(default_factory=lambda: R.get("figure", "share"))
    panel_size: tuple = field(default_factory=lambda: tuple(R.get("figure", "panel_size")))
    legend: str = field(default_factory=lambda: R.get("legend", "where"))
    legend_loc: str = field(default_factory=lambda: R.get("legend", "loc"))
    legend_ncol: int = field(default_factory=lambda: int(R.get("legend", "ncol")))
    titles: str = "auto"
    label_mode: str = "all"
    sublabels: bool = False
    row_captions: str = "off"
    suptitle: str | None = None
    xscale: float = field(default_factory=lambda: float(R.get("figure", "xscale")))
    ylim: tuple | None = None
    logy: bool = False
    ylabel: str | None = None            # None = quella della metrica
    # ritocchi fatti a mano nell'anteprima: etichetta di partenza -> {name, color}.
    # Valgono per questa figura soltanto; per renderli permanenti si incolla la
    # regola [[series]] in style.toml (il selettore la scrive gia' pronta).
    series_overrides: dict = field(default_factory=dict)

    # --- serializzazione -----------------------------------------------------

    def to_dict(self) -> dict:
        d = asdict(self)
        d["version"] = SPEC_VERSION
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "FigureSpec":
        data = dict(data or {})
        data.pop("version", None)
        known = {f.name for f in dataclass_fields(cls)}
        base = data.pop("baseline", None) or {}
        if isinstance(base, str):          # forma vecchia: solo la famiglia
            base = {"family": base}
        spec = cls(**{k: v for k, v in data.items() if k in known})
        spec.baseline = BaselineSpec(**{k: v for k, v in base.items()
                                        if k in {"filters", "family", "epochs"}})
        if spec.ylim:
            spec.ylim = tuple(spec.ylim)
        if spec.panel_size:
            spec.panel_size = tuple(spec.panel_size)
        return spec

    # --- opzioni di disegno --------------------------------------------------

    def grid_options(self, hue, ylabel: str) -> GridOptions:
        return GridOptions(
            rows=self.rows, cols=self.cols, xscale=self.xscale, xmax=self.xmax,
            ylim=self.ylim, share=self.share, panel_size=tuple(self.panel_size),
            legend=self.legend, legend_loc=self.legend_loc, legend_ncol=self.legend_ncol,
            titles=self.titles, label_mode=self.label_mode, sublabels=self.sublabels,
            row_captions=self.row_captions, suptitle=self.suptitle, ylabel=ylabel,
            logy=self.logy, paper=self.paper, hue=list(hue),
        )


# --- scelta delle serie -----------------------------------------------------

def _series_styles(agg, ckey_colors: dict) -> dict:
    """Etichetta -> come si disegna quella curva.

    Le regole `[[series]]` di style.toml vincono su tutto; quello che non
    dicono viene dalla palette (colore) e dalla sezione `[lines]`.
    """
    out = {}
    for rec in agg.drop_duplicates("label").to_dict("records"):
        rule = R.rule_for(rec)
        out[rec["label"]] = {
            "color": rule.get("color") or ckey_colors[rec["ckey"]],
            "width": float(rule.get("width", S.line_width())),
            "style": rule.get("style", "solid"),
            "band_alpha": float(rule.get("band_alpha", S.band_alpha())),
            "latex": rule.get("latex"),
        }
    return out


def _apply_overrides(agg, order, styles, matches, overrides: dict):
    """Applica i ritocchi fatti a mano nell'anteprima (nome, colore, tratteggio).

    Sono indicizzati per etichetta *di partenza*, quella calcolata dai dati: cosi'
    rinominare non fa perdere il collegamento, e cambiare i colori o i filtri non
    sposta il ritocco su un'altra curva. Una serie che non c'e' piu' viene
    ignorata in silenzio — la selezione e' cambiata, non c'e' niente da fare.
    Vale sia per le serie normali sia per le baseline: chiamata due volte con
    due `styles` distinti, uno per gruppo (vedi `_baseline_blocks`).
    """
    overrides = {k: v for k, v in (overrides or {}).items() if k in styles}
    if not overrides:
        return agg, order, styles, matches
    renames = {}
    for original, over in overrides.items():
        new = (over.get("name") or "").strip() or original
        style = dict(styles.pop(original))
        if over.get("color"):
            style["color"] = over["color"]
        if over.get("style"):
            style["style"] = over["style"]
        if new != original:
            # rinominando a mano vince il nome scelto, anche nel .tex: tenere la
            # macro della regola farebbe uscire un'etichetta diversa da quella
            # dell'anteprima. Per una macro si scrive `latex` in style.toml.
            style["latex"] = None
        styles[new] = style
        matches[new] = matches.pop(original, {})
        renames[original] = new
    agg = agg.copy()
    agg["label"] = agg.label.map(lambda lab: renames.get(lab, lab))
    order = [renames.get(lab, lab) for lab in order]
    return agg, order, styles, matches


def auto_hue(df, exclude=()) -> list:
    """Serie = tutte le dimensioni di ablation che variano nella selezione.

    Cosi' due configurazioni diverse non finiscono mai mediate nella stessa
    curva: l'unica cosa su cui si aggrega sono i seed. Le dimensioni gia'
    assegnate a righe/colonne sono escluse.
    """
    exclude = {c for c in exclude if c}
    hue = [c for c in schema.SERIES_FIELDS
           if c in df.columns and c not in exclude and df[c].nunique(dropna=False) > 1]
    if not hue:
        return ["family"]
    # Toglie le dimensioni ridondanti (determinate da un'altra gia' presente, come
    # adaptive_lr rispetto a family): non separano nulla, sporcano solo la legenda.
    n = df.groupby(hue, dropna=False).ngroups
    for col in reversed(list(hue)):
        rest = [c for c in hue if c != col]
        if rest and df.groupby(rest, dropna=False).ngroups == n:
            hue = rest
    return hue


def merged_dims(df, hue, panels=()) -> list:
    """Dimensioni che variano ma non separano le curve: finiscono mediate."""
    covered = set(hue) | {c for c in panels if c}
    return [c for c in schema.SERIES_FIELDS
            if c in df.columns and c not in covered and df[c].nunique(dropna=False) > 1]


def _sort_ascending(df, cols) -> list:
    """I booleani (es. opc) vanno decrescenti: la variante 'piena' prima di 'Off'."""
    return [not (df[c].dtype == bool or set(df[c].dropna().unique()) <= {True, False})
            for c in cols]


# --- baseline ---------------------------------------------------------------

def _baseline_blocks(full_index, sel, spec: FigureSpec, panel_fields, metric):
    """DataFrame aggregato delle baseline (con colonne `color`/`style`), o None.

    Le baseline richieste possono essere piu' d'una (`family` e' un elenco, o i
    filtri ne pescano diverse): ognuna e' un blocco a se', con il proprio
    tratteggio di default, cosi' restano distinguibili pur essendo tutte nere —
    ma nome, colore e tratteggio si possono ritoccare a mano come per le serie
    normali (`spec.series_overrides`, stessa chiave: l'etichetta).
    """
    conf = spec.baseline
    if not conf.active() or full_index is None:
        return None
    if conf.filters:
        pool = select_runs(full_index, conf.filters, state=spec.state)
    else:
        pool = full_index[full_index.family.isin(conf.families())
                          & (full_index.state == "finished")]
    # Ristrette agli environment della selezione: altrimenti in ogni pannello
    # finiscono baseline di environment diversi.
    if "env" in sel.columns and sel.env.notna().any() and "env" in pool.columns:
        pool = pool[pool.env.isin(sel.env.dropna().unique())]
    if pool.empty:
        print("[plot] attenzione: nessun run di baseline trovato")
        return None

    def block(runs, style: str, label_suffix: str = "", fixed=None):
        if runs.empty:
            return None
        curves = load_curves(runs, source=spec.source, metric=metric, verbose=False)
        if curves.empty:
            return None
        # Le baseline non variano lungo omega/setting/IS: si aggregano su
        # famiglia ed environment e poi si replicano sui pannelli.
        agg = aggregate(curves, runs, ["env", "family"], band=spec.band,
                        smooth=spec.smooth, grid_points=spec.grid_points, xmax=spec.xmax)
        agg["label"] = [S.mathtt(L.family_name({"family": f}, paper=spec.paper))
                        + label_suffix for f in agg["family"]]
        agg["color"] = S.baseline_color()
        agg["style"] = style
        for f in panel_fields:
            if fixed and f in fixed:
                agg[f] = fixed[f]          # vale per un pannello solo
            elif f not in agg.columns:
                vals = sorted(sel[f].dropna().unique()) if f in sel.columns else []
                if vals:                   # replicata su tutti i pannelli
                    agg = agg.merge(pd.DataFrame({f: vals}), how="cross")
        return agg

    # Ordine: quello scelto nella pagina; con i filtri, quello dell'indice.
    found = list(dict.fromkeys(pool.family.dropna()))
    fams = [f for f in conf.families() if f in set(found)] or found
    styles = S.baseline_styles()
    blocks = []
    for i, fam in enumerate(fams):
        fam_pool = pool[pool.family == fam]
        # Le PPO delle ablation sul clip hanno epoch_mult=1 come la baseline
        # vera: senza escluderle finirebbero mediate dentro di essa. Dove invece
        # la famiglia esiste *solo* dentro un'ablation — GePPO-original sta
        # tutto sotto `ablation=geppo_original` — non c'e' niente da escludere.
        if "ablation" in fam_pool.columns:
            clean = fam_pool[fam_pool.ablation.isna()]
            fam_pool = clean if not clean.empty else fam_pool
        # Il riferimento e' la variante a epoche base: senza questa riga B2 e B3
        # (x1 e x2/x4/x8) finiscono mediati in un'unica linea nera.
        base_pool = fam_pool
        if "epoch_mult" in fam_pool.columns and fam_pool.epoch_mult.nunique(dropna=False) > 1:
            at_one = fam_pool[fam_pool.epoch_mult == 1]
            base_pool = at_one if not at_one.empty else fam_pool
        blocks.append(block(base_pool, styles[i % len(styles)]))

        extra = str(conf.epochs or "")
        if extra == "follow_window" and "window" in sel.columns:
            for w in sorted(w for w in sel.window.dropna().unique() if w > 1):
                blocks.append(block(fam_pool[fam_pool.epoch_mult == w], "dashed",
                                    r" ($\omega \tilde{K}$ epochs)", {"window": w}))
        elif extra not in ("", "none"):
            mult = float(extra)
            blocks.append(block(fam_pool[fam_pool.epoch_mult == mult], "dashed",
                                rf" ($\times {int(mult)}$ epochs)"))

    blocks = [b for b in blocks if b is not None]
    if not blocks:
        return None
    agg = pd.concat(blocks, ignore_index=True)
    return _apply_baseline_overrides(agg, spec.series_overrides)


def _apply_baseline_overrides(agg, overrides: dict):
    """Ritocchi a mano (nome/colore/tratteggio) sulle etichette delle baseline.

    Stesso meccanismo delle serie normali (`_apply_overrides`), ma qui `styles`
    si costruisce dalle colonne gia' calcolate (`color`, `style`) invece che
    dalle regole di `style.toml`, che le baseline non consultano.
    """
    if not overrides:
        return agg
    styles = {rec["label"]: {"color": rec["color"], "style": rec["style"]}
              for rec in agg.drop_duplicates("label").to_dict("records")}
    agg, _, styles, _ = _apply_overrides(agg, [], styles, {}, overrides)
    for lab, st in styles.items():
        mask = agg.label == lab
        agg.loc[mask, "color"] = st["color"]
        agg.loc[mask, "style"] = st["style"]
    return agg


# --- pipeline ---------------------------------------------------------------

@dataclass
class Series:
    """Risultato della preparazione: i dati e come vanno disegnati."""

    sel: pd.DataFrame           # run selezionati
    agg: pd.DataFrame           # curve aggregate sui seed
    order: list                 # ordine delle serie in legenda
    styles: dict                # etichetta -> {color, width, style, band_alpha, latex}
    matches: dict               # etichetta -> valori che la identificano (per style.toml)
    hue: list                   # dimensioni che decidono il colore
    baseline: pd.DataFrame | None
    ylabel: str
    metric_label: str
    merged: list                # dimensioni che variano senza separare le curve


def select(index: pd.DataFrame, spec: FigureSpec) -> pd.DataFrame:
    """Run che la figura deve usare (selezione esplicita + filtri + stato)."""
    df = index[index.run_id.isin(spec.run_ids)] if spec.run_ids is not None else index
    return select_runs(df, spec.filters, state=spec.state)


def prepare(index: pd.DataFrame, spec: FigureSpec,
            full_index: pd.DataFrame | None = None, verbose: bool = True) -> Series:
    """Selezione -> curve -> aggregazione -> etichette, colori, baseline."""
    full_index = index if full_index is None else full_index
    sel = select(index, spec)
    if sel.empty:
        raise ValueError("Nessun run corrisponde ai filtri.")

    info = metric_info(spec.metric)
    hue = [c for c in (spec.hue or []) if c in sel.columns]
    hue = hue or auto_hue(sel, exclude=(spec.rows, spec.cols))
    merged = merged_dims(sel, hue, (spec.rows, spec.cols))
    if verbose:
        print(f"[plot] {len(sel)} run selezionati; serie per: {', '.join(hue)}")
        if merged:
            print(f"[plot] ATTENZIONE: {', '.join(merged)} variano ma non separano le "
                  f"curve: configurazioni diverse finiscono mediate insieme "
                  f"(togli --hue per le serie automatiche)")

    group_cols = sorted(set(hue) | {"family"} | {c for c in (spec.rows, spec.cols) if c})
    curves = load_curves(sel, source=spec.source, metric=spec.metric, verbose=verbose)
    if curves.empty:
        raise ValueError(f"Nessun dato per «{info['label']}» in questa selezione.")
    agg = aggregate(curves, sel, group_cols, band=spec.band, smooth=spec.smooth,
                    grid_points=spec.grid_points, xmax=spec.xmax)
    agg = agg[agg.n_seeds >= spec.min_seeds].copy()
    if agg.empty:
        raise ValueError(f"Nessuna serie con almeno {spec.min_seeds} seed.")

    # Etichette: per default le colonne di hue; label_fields puo' aggiungerne
    # altre (es. window, per avere «ω = 4» in legenda come nel paper).
    label_fields = tuple(spec.label_fields or hue) + ("family",)
    agg["label"] = [L.series_label(r, fields=label_fields, paper=spec.paper)
                    for r in agg.to_dict("records")]
    # Il colore dipende solo da hue: cosi' resta lo stesso in tutti i pannelli
    # anche se l'etichetta contiene la dimensione di riga/colonna.
    hue_cols = [c for c in hue if c in agg.columns]
    agg["ckey"] = agg[hue_cols].astype(str).agg("|".join, axis=1)
    hue_keys = (agg.drop_duplicates("ckey")
                   .sort_values(hue_cols, ascending=_sort_ascending(agg, hue_cols))["ckey"]
                   .tolist())
    ckey_colors = dict(zip(hue_keys, S.color_cycle(len(hue_keys))))
    styles = _series_styles(agg, ckey_colors)
    sort_cols = hue_cols + [c for c in (spec.rows, spec.cols)
                            if c and c in agg.columns and c not in hue_cols]
    order = (agg.drop_duplicates("label")
                .sort_values(sort_cols, ascending=_sort_ascending(agg, sort_cols))
                ["label"].tolist())
    if spec.hue_order:
        order = [o for o in spec.hue_order if o in set(agg.label)]

    # match delle serie: da qui esce la regola [[series]] da incollare in style.toml
    matches = {rec["label"]: {c: rec.get(c) for c in hue_cols}
               for rec in agg.drop_duplicates("label").to_dict("records")}
    agg, order, styles, matches = _apply_overrides(
        agg, order, styles, matches, spec.series_overrides)

    panel_fields = [c for c in (spec.rows, spec.cols) if c]
    base_agg = _baseline_blocks(full_index, sel, spec, panel_fields, spec.metric)

    # etichetta y: quella dello spec, poi quella scritta in style.toml, infine
    # quella della metrica scelta
    ylabel = spec.ylabel or R.get("figure", "ylabel") or info["ylabel"]
    return Series(sel=sel, agg=agg, order=order, styles=styles, hue=hue,
                  matches=matches, baseline=base_agg, ylabel=ylabel,
                  metric_label=info["label"], merged=merged)


def draw(series: Series, spec: FigureSpec):
    """Figura matplotlib a partire da un `Series` gia' preparato."""
    return draw_grid(series.agg, series.order, series.styles,
                     spec.grid_options(series.hue, series.ylabel), series.baseline)


def build(index: pd.DataFrame, spec: FigureSpec,
          full_index: pd.DataFrame | None = None, verbose: bool = True):
    """(figura, Series) — il percorso completo, uguale per CLI e selettore."""
    series = prepare(index, spec, full_index=full_index, verbose=verbose)
    return draw(series, spec), series


@dataclass
class Panel:
    """Un riquadro della griglia, disegnabile da solo."""

    series: Series
    spec: FigureSpec
    fixed: dict          # dimensione -> valore che identifica il pannello
    row: int
    col: int

    @property
    def slug(self) -> str:
        """Suffisso per il nome file: env_reacher_setting_2 (vuoto se unico)."""
        bits = [f"{c}_{_slug_value(v)}" for c, v in self.fixed.items()]
        return "_".join(bits)

    @property
    def caption(self) -> str:
        """Cosa fissa questo pannello, in LaTeX.

        `panel_title` produce mathtext ($\\omega = 2$), che e' gia' LaTeX valido:
        e' lo stesso titolo che il pannello aveva dentro la griglia.
        """
        return ", ".join(schema.panel_title(c, v) for c, v in self.fixed.items())


def _slug_value(value) -> str:
    text = str(value).replace("-v5", "").replace(".0", "")
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_").lower()


def split_panels(series: Series, spec: FigureSpec) -> list[Panel]:
    """La griglia spacchettata in pannelli indipendenti, uno per figura.

    Serve a esportare un `.tex` per riquadro: in un paper i pannelli si
    compongono in LaTeX, quindi l'export deve dare i pezzi, non il mosaico.
    Colori, ordine delle serie ed etichette restano quelli calcolati sull'intera
    griglia — un pannello estratto e' identico a come si vedeva nell'anteprima.
    """
    from .grid import panel_values

    row_vals = panel_values(series.agg, spec.rows)
    col_vals = panel_values(series.agg, spec.cols)
    # da solo un pannello non ha ne' righe ne' colonne su cui titolare, e la
    # caption di riga la scrive LaTeX
    flat = dataclass_replace(spec, rows=None, cols=None, suptitle=None,
                             row_captions="off", sublabels=False, label_mode="all")
    panels = []
    for i, rv in enumerate(row_vals):
        for j, cv in enumerate(col_vals):
            fixed = {}
            agg, base = series.agg, series.baseline
            for col, value in ((spec.rows, rv), (spec.cols, cv)):
                if not col:
                    continue
                fixed[col] = value
                agg = agg[agg[col] == value]
                if base is not None and col in base.columns and base[col].notna().any():
                    base = base[base[col] == value]
            if agg.empty:
                continue
            order = [lab for lab in series.order if lab in set(agg.label)]
            panels.append(Panel(
                series=dataclass_replace(series, agg=agg, baseline=base, order=order),
                spec=flat, fixed=fixed, row=i, col=j))
    return panels


def n_panels(series: Series, spec: FigureSpec) -> int:
    rows = series.agg[spec.rows].nunique() if spec.rows else 1
    cols = series.agg[spec.cols].nunique() if spec.cols else 1
    return max(1, rows) * max(1, cols)
