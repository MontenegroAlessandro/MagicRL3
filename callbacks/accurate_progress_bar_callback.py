from stable_baselines3.common.callbacks import BaseCallback

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


class AccurateProgressBarCallback(BaseCallback):
    """
    Drop-in replacement for SB3's built-in `progress_bar=True` (ProgressBarCallback),
    which advances the bar by `training_env.num_envs` on every `on_step()` call --
    silently assuming one `on_step()` call always corresponds to exactly `num_envs`
    real environment timesteps. That assumption breaks for algorithms like FDPG,
    which add extra timesteps (e.g. perturbed rollouts) directly to `num_timesteps`
    without ever calling `on_step()` for them -- so the built-in bar only reflects
    the reference rollout and silently jumps from ~50% straight to closed when
    training actually finishes.

    This callback instead tracks `num_timesteps` itself (resynced from the model at
    training end, since it can otherwise lag behind by one un-on_step()'d batch),
    so the bar's progress is accurate for both regular and FDPG-style algorithms.
    """

    def __init__(self) -> None:
        super().__init__()
        if tqdm is None:
            raise ImportError(
                "You must install tqdm in order to use the progress bar callback."
            )
        self.pbar = None

    def _on_training_start(self) -> None:
        self.pbar = tqdm(total=self.locals["total_timesteps"] - self.model.num_timesteps)

    def _on_step(self) -> bool:
        self.pbar.update(self.num_timesteps - self.pbar.n)
        return True

    def _on_training_end(self) -> None:
        # self.num_timesteps is only synced from the model inside on_step(); resync here
        # so timesteps added outside of on_step() (e.g. FDPG's perturbed rollouts) aren't
        # missed if they happened after the last on_step() call.
        self.num_timesteps = self.model.num_timesteps
        self.pbar.update(self.num_timesteps - self.pbar.n)
        self.pbar.refresh()
        self.pbar.close()
