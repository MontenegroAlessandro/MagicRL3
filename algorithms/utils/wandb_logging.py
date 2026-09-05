"""Forward SB3 dumps to W&B with separate environment and diagnostic axes."""

from numbers import Number

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import KVWriter


class WandbMetricsWriter(KVWriter):
    """Commit complete rows; never use either training clock as W&B's step.

    TensorBoard synchronization must be disabled for this run to avoid duplicate
    ingestion. The existing time/total_timesteps key supplies SB3's plotting axis.
    """

    def __init__(self, run):
        self.run = run
        run.define_metric("diag/*", step_metric="diag/time/flat_step", step_sync=False)
        for prefix in ("train", "eval", "rollout", "time"):
            run.define_metric(f"{prefix}/*", step_metric="time/total_timesteps", step_sync=False)

    def write(self, key_values, key_excluded, step=0):
        diagnostics = {}
        standard = {}
        for key, value in key_values.items():
            # Preserve the metric selection of the TensorBoard integration.
            if "tensorboard" in key_excluded.get(key, ()) or "wandb" in key_excluded.get(key, ()):
                continue
            if isinstance(value, np.generic):
                value = value.item()
            if not isinstance(value, Number):
                continue
            if key.startswith("diag/"):
                diagnostics[key] = value
            else:
                standard[key] = value

        if standard:
            standard["time/total_timesteps"] = step
            self.run.log(standard, commit=True)
        if diagnostics:
            # Supplied in every POSER snapshot, without reusing a previous row's axis.
            if "diag/time/flat_step" not in diagnostics:
                raise ValueError("POSER diagnostics require diag/time/flat_step")
            self.run.log(diagnostics, commit=True)

    def close(self):
        # The runner owns the run and calls finish().
        pass


class WandbMetricsCallback(BaseCallback):
    """Attach the writer after SB3 configures its logger for learn()."""

    def __init__(self, run):
        super().__init__()
        self.writer = WandbMetricsWriter(run)

    def _on_training_start(self):
        self.logger.output_formats.append(self.writer)

    def _on_step(self):
        return True

    def _on_training_end(self):
        self.logger.output_formats.remove(self.writer)
