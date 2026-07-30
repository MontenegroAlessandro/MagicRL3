"""Indice dei run: metadati (una riga per run) letti da W&B e messi in cache.

La cache sta in /storage (INDEX_PARQUET). L'aggiornamento e' incrementale: la
lista dei run (id/nome/stato/tag) viene sempre riscaricata, la config completa
solo per i run non ancora in cache o rimasti non finiti. Le run cancellate da
W&B escono dall'indice al primo aggiornamento (la lista viva fa da riferimento).

Come si legge una run non sta qui ma in `sources/`: una fonte per gruppo di
progetti con convenzioni proprie (`sources/current.py`, `sources/paper.py`).
Questo modulo si occupa solo di scaricare, unire e mettere in cache.

Colonne prodotte:
  run_id, name, group, state, tags, created_at, campaign, dir_name,
  project, source, ablation,
  family (PPO|RT-PPO|GePPO|SAC|TD3), env, seed, window, is_type (N|BH),
  setting (1|2|3), fresh_adv, opc, adaptive_lr, sampling, seq,
  n_steps, batch_size, n_minibatch, lr, gamma, total_timesteps, eval_freq,
  n_epochs, epoch_mult, clip_range
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from . import sources
from .paths import INDEX_CSV, INDEX_PARQUET, ensure_dirs, wandb_path


def _warn_homonyms(df: pd.DataFrame, verbose: bool) -> None:
    """Segnala run omonime nei progetti dove il nome identifica l'esperimento.

    Nei progetti del paper il nome codifica tutti i parametri piu' il seed: due
    run omonime sono lo stesso esperimento lanciato due volte, e mediarle
    entrambe restringe la banda senza cambiare la media. I doppioni sono stati
    ripuliti a monte su W&B, quindi qui non si scarta piu' niente in automatico:
    se ne ricompaiono, meglio saperlo e decidere, che vederli sparire in silenzio.

    Nelle campagne nuove il nome copre solo una parte dei parametri (le run
    Swimmer `random` e `balanced` sono omonime ma diverse), quindi li' l'omonimia
    non vuol dire niente e l'avviso non scatta.
    """
    if not verbose or "name" not in df.columns or "project" not in df.columns:
        return
    keyed = [p for p, src in sources.BY_PROJECT.items() if src.name_is_key]
    sub = df[df.project.isin(keyed)]
    if "ablation" in sub.columns:
        sub = sub[sub.ablation.isna()]
    dup = sub[sub.duplicated(["project", "name"], keep=False)]
    if not dup.empty:
        print(f"[index] ATTENZIONE: {len(dup)} run omonime nello stesso progetto "
              f"(es. {dup.name.iloc[0]!r}): sono rilanci dello stesso esperimento? "
              f"se lo sono gonfiano n_seeds")


def build_index(force: bool = False, workers: int = 8,
                projects: list[str] | None = None, verbose: bool = True) -> pd.DataFrame:
    """(Ri)costruisce l'indice dei run su tutti i progetti e lo salva in cache."""
    import wandb

    ensure_dirs()
    projects = list(projects or sources.ALL_PROJECTS)
    cached = pd.DataFrame()
    if INDEX_PARQUET.exists() and not force:
        cached = pd.read_parquet(INDEX_PARQUET)

    api = wandb.Api()
    runs = []  # (run, progetto)
    for project in projects:
        found = list(api.runs(wandb_path(project), per_page=200))
        runs += [(r, project) for r in found]
        if verbose:
            print(f"[index] {len(found)} run su {project}")

    known = set()
    if not cached.empty:
        # I run non finiti vanno riletti: la config cambia fino alla fine.
        known = set(cached.loc[cached.state == "finished", "run_id"])
    todo = [t for t in runs if t[0].id not in known]
    if verbose:
        print(f"[index] config da scaricare: {len(todo)} (in cache: {len(known)})")

    def fetch(item):
        run, project = item
        try:
            # Indispensabile: nell'elenco la config dei progetti del paper e' vuota.
            run.load(force=True)
            return sources.for_run(run.tags, project).row(run, project)
        except Exception as exc:  # run corrotto o rimosso: non blocca l'indice
            return dict(run_id=run.id, name=run.name, state=run.state, project=project,
                        tags=",".join(run.tags or []), error=str(exc))

    rows = []
    if todo:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, row in enumerate(pool.map(fetch, todo), 1):
                rows.append(row)
                if verbose and i % 100 == 0:
                    print(f"[index]   {i}/{len(todo)}")

    new = pd.DataFrame(rows)
    df = pd.concat([cached, new], ignore_index=True) if not cached.empty else new
    df = df.drop_duplicates(subset="run_id", keep="last")

    # stato/tag sempre aggiornati dalla lista (economici). Il join a destra fa
    # anche da pulizia: quello che non c'e' piu' su W&B esce dall'indice. Vale
    # solo per i progetti appena riletti: gli altri restano in cache come sono,
    # altrimenti --projects cancellerebbe il resto dell'indice.
    live = pd.DataFrame([{"run_id": r.id, "state": r.state,
                          "tags": ",".join(r.tags or [])} for r, _ in runs])
    other = df[~df.project.isin(projects)]
    fresh = (df[df.project.isin(projects)]
             .drop(columns=["state", "tags"]).merge(live, on="run_id", how="right"))
    df = pd.concat([other, fresh], ignore_index=True) if not other.empty else fresh
    _warn_homonyms(df, verbose)

    df.to_parquet(INDEX_PARQUET, index=False)
    df.to_csv(INDEX_CSV, index=False)
    if verbose:
        print(f"[index] scritto {INDEX_PARQUET} ({len(df)} run)")
    return df


def load_index(auto_build: bool = True) -> pd.DataFrame:
    """Carica l'indice dalla cache (costruendolo se manca)."""
    if not INDEX_PARQUET.exists():
        if not auto_build:
            raise FileNotFoundError(
                f"Indice assente: {INDEX_PARQUET}. Lancia plots/scripts/build_index.py"
            )
        return build_index()
    return pd.read_parquet(INDEX_PARQUET)
