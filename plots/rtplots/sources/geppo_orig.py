"""Run della codebase originale di GePPO (tag `geppo_original`).

Convenzione diversa da tutto il resto: la config non ha `experiment` ma chiavi
piatte `<gruppo>/<nome>` (`env_kwargs/env_name`, `runner_kwargs/M`,
`ac_kwargs/eps_ppo`, ...), quindi serve una fonte a se'.

Vive nello stesso progetto W&B delle campagne correnti: il riconoscimento e'
quindi per **tag** (`claim_tags`), non per progetto.

Scelte di mappatura, decise a mano:
  - `family = "GePPO-original"`: serie distinta dal nostro GePPO;
  - `window = runner_kwargs/M` (la finestra di policy passate), non M*B;
  - `setting = None`: queste configurazioni sono tunate e non appartengono alla
    griglia 1/2/3, quindi non si deduce niente;
  - `ablation = "geppo_original"`: stanno fuori dai sottospazi di STATUS_EXP,
    percio' `ablation=none` continua a isolare P1+P2+B2+B3 come prima;
  - `fresh_adv`, `opc`, `sampling`, `seq`, `is_type` restano vuoti: nella
    codebase originale non esistono come opzioni, e un valore inventato
    sarebbe indistinguibile da uno vero.

Curve: nessun `evaluations.npz` locale (il layout dei log e' un altro), quindi
`eval/mean_reward` arriva dalla history W&B, che c'e' con lo stesso nome.
"""
from __future__ import annotations

from pathlib import Path

from .base import RunSource, _num, as_bool

CLAIM_TAG = "geppo_original"


class GePPOOriginalSource(RunSource):
    def row(self, run, project: str) -> dict:
        cfg = run.config or {}
        tags = list(run.tags or [])
        env = cfg.get("env_kwargs/env_name")
        n_steps = cfg.get("runner_kwargs/n")
        n_minibatch = cfg.get("ac_kwargs/nminibatch")
        batch_size = n_steps // n_minibatch if n_steps and n_minibatch else None
        n_epochs = cfg.get("ac_kwargs/update_it")
        save_path = cfg.get("setup_kwargs/save_path") or ""
        dir_name = save_path if save_path.startswith("/") else None
        # il tag numerato e' il nome della campagna anche quando save_path e'
        # relativo ("./logs"), com'e' nelle prime run
        campaign = next((t for t in tags if t[:2].isdigit() and t[2:3] == "_"), None)
        return dict(
            run_id=run.id,
            name=run.name,
            group=run.group,
            state=run.state,
            tags=",".join(tags),
            created_at=str(run.created_at),
            project=project,
            source=self.name,
            ablation=CLAIM_TAG,
            dir_name=dir_name,
            campaign=campaign or (Path(dir_name).name if dir_name else None),
            family="GePPO-original",
            env=env,
            seed=cfg.get("seed"),
            window=cfg.get("runner_kwargs/M") or 1,
            is_type=None,
            setting=None,
            fresh_adv=None,
            opc=None,
            adaptive_lr=as_bool(cfg.get("ac_kwargs/adapt_lr")),
            sampling=None,
            seq=None,
            n_steps=n_steps,
            batch_size=batch_size,
            n_minibatch=n_minibatch,
            lr=cfg.get("ac_kwargs/actor_lr"),
            gamma=cfg.get("runner_kwargs/gamma"),
            total_timesteps=_num(cfg.get("train_kwargs/sim_size")),
            eval_freq=cfg.get("eval_kwargs/eval_freq"),
            n_epochs=n_epochs,
            epoch_mult=self.epoch_mult(env, n_epochs),
            clip_range=cfg.get("ac_kwargs/eps_ppo"),
        )


SOURCE = GePPOOriginalSource(
    name="geppo-orig",
    claim_tags=frozenset({CLAIM_TAG}),
    # il nome codifica tutti i parametri piu' il seed
    name_is_key=True,
)
