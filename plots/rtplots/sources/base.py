"""Come si legge una run W&B: parte comune a tutte le fonti.

Una `RunSource` descrive un gruppo di progetti che condividono convenzioni:
quali progetti sono, quali tag hanno significato, quali default valgono per le
chiavi assenti dalla config, e con quale nome ciascuna metrica del catalogo e'
loggata la' dentro.

Aggiungere una campagna con convenzioni nuove = aggiungere un file in
`sources/`, senza toccare `index.py`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..envbase import base_for


def as_bool(v):
    if isinstance(v, str):
        return v.lower() in ("true", "1", "yes")
    return bool(v) if v is not None else None


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


@dataclass
class RunSource:
    """Convenzioni di lettura di un gruppo di progetti W&B."""

    name: str                                   # valore della colonna `source`
    projects: dict[str, str] = field(default_factory=dict)   # progetto -> env atteso
    # tag che assegnano una run a questa fonte anche se il progetto e' di
    # un'altra: serve dove convenzioni diverse convivono nello stesso progetto
    claim_tags: frozenset[str] = frozenset()
    # tag -> setting, per le fonti che non usano i tag setting1/2/3
    setting_tags: dict[str, int] = field(default_factory=dict)
    # tag che marcano run fuori dai sottospazi di STATUS_EXP
    ablation_tags: tuple[str, ...] = ()
    # valori veri per chiavi che nella config non esistono (non "ignoto")
    defaults: dict = field(default_factory=dict)
    # chiave del catalogo -> chiave davvero loggata in questa fonte
    metric_aliases: dict[str, str] = field(default_factory=dict)
    # metriche del catalogo che qui non esistono con la stessa semantica:
    # chiedendole si ottiene un avviso esplicito invece di run che spariscono
    metrics_unavailable: dict[str, str] = field(default_factory=dict)
    # il nome della run identifica l'esperimento (tutti i parametri piu' il seed)?
    # Dove e' vero, due run omonime nello stesso progetto sono un rilancio dello
    # stesso esperimento e vanno segnalate. Dove e' falso il nome ne copre solo
    # una parte: le campagne Swimmer random e balanced, per dire, hanno nomi
    # identici ed esperimenti diversi (differiscono per `sampling`).
    name_is_key: bool = False

    # --- lettura di una run --------------------------------------------------

    def row(self, run, project: str) -> dict:
        """Riga dell'indice per una run di questa fonte."""
        cfg = run.config or {}
        exp = dict(cfg.get("experiment") or {})
        for key, value in self.defaults.items():
            exp.setdefault(key, value)
        tags = list(run.tags or [])
        dir_name = exp.get("dir_name")
        window = exp.get("window_size") or 1
        n_steps = exp.get("n_steps")
        batch_size = exp.get("batch_size") or cfg.get("batch_size")
        n_envs = exp.get("n_envs") or 1
        n_minibatch = None
        if batch_size:
            n_minibatch = (n_steps or 0) * n_envs * window // batch_size or None
        weight = (exp.get("weight_type") or "").lower()
        env = exp.get("env_name") or self.projects.get(project)
        return dict(
            run_id=run.id,
            name=run.name,
            group=run.group,
            state=run.state,
            tags=",".join(tags),
            created_at=str(run.created_at),
            project=project,
            source=self.name,
            ablation=next((t for t in self.ablation_tags if t in tags), None),
            dir_name=dir_name,
            campaign=Path(dir_name).name if dir_name else None,
            family=self.family(cfg.get("algo"), exp),
            env=env,
            seed=exp.get("seed"),
            window=window,
            is_type={"bh": "BH", "naive": "N", "mpm": "MPM"}.get(
                weight, weight.upper() or None),
            setting=self.setting(exp, tags, env),
            fresh_adv=as_bool(exp.get("fresh_adv")),
            opc=as_bool(exp.get("on_policy_critic")),
            adaptive_lr=as_bool(exp.get("adaptive_lr")),
            sampling=self.sampling(exp),
            n_steps=n_steps,
            batch_size=batch_size,
            n_minibatch=n_minibatch,
            lr=exp.get("learning_rate"),
            gamma=exp.get("gamma"),
            total_timesteps=_num(exp.get("total_timesteps")),
            eval_freq=exp.get("eval_freq"),
            n_epochs=exp.get("n_epochs"),
            epoch_mult=self.epoch_mult(env, exp.get("n_epochs")),
            clip_range=exp.get("clip_range"),
        )

    # --- pezzi sovrascrivibili dalle singole fonti ---------------------------

    @staticmethod
    def family(algo: str | None, exp: dict) -> str:
        if algo in ("SAC", "TD3"):
            return algo
        if as_bool(exp.get("geppo_clip")):
            return "GePPO"
        return "RT-PPO" if (exp.get("window_size") or 1) > 1 else "PPO"

    @staticmethod
    def sampling(exp: dict) -> str | None:
        """random | balanced | weighted.

        La convenzione attuale e' `batch_sampling` (stringa, tre valori). Le run
        piu' vecchie hanno il booleano `balanced_batches`: si mappa qui, cosi' le
        due generazioni finiscono sullo stesso asse invece di lasciare le nuove
        senza valore.
        """
        value = exp.get("batch_sampling")
        if isinstance(value, str) and value:
            return value.lower()
        legacy = as_bool(exp.get("balanced_batches"))
        if legacy is None:
            return None
        return "balanced" if legacy else "random"

    def setting(self, exp: dict, tags: list[str], env: str | None):
        """setting 1/2/3: dal tag se c'e', altrimenti dedotto dai parametri."""
        for t in tags:
            if t.startswith("setting") and t[7:].isdigit():
                return int(t[7:])
            if t in self.setting_tags:
                return self.setting_tags[t]
        if (exp.get("window_size") or 1) <= 1:
            return None
        base = base_for(env)
        if not base:
            return None
        window = exp["window_size"]
        if exp.get("n_steps") == base["n_steps"] // window:
            return 3
        if exp.get("batch_size") == base["batch_size"]:
            return 1
        return 2

    @staticmethod
    def epoch_mult(env: str | None, n_epochs):
        """Epoche in multipli del valore base dell'environment: 1, 2, 4, 8.

        E' cio' che distingue le due baseline PPO di STATUS_EXP.md: B2 e' il PPO
        base (`epoch_mult=1`), B3 sono le varianti a epoche x2, x4, x8. Senza
        questa colonna le quattro celle PPO di un environment differiscono solo
        per `n_epochs` e in figura collassano in un'unica serie.
        """
        base = base_for(env).get("n_epochs")
        if not base or not n_epochs:
            return None
        mult = n_epochs / base
        return int(mult) if float(mult).is_integer() else round(mult, 3)

    # --- metriche ------------------------------------------------------------

    def metric_key(self, key: str) -> str | None:
        """Chiave davvero loggata in questa fonte, o None se non esiste qui."""
        if key in self.metrics_unavailable:
            return None
        return self.metric_aliases.get(key, key)
