python3 run.py \
    wandb.project=fd \
    algo@experiment.algo=fdpg \
    experiment.env_name=Swimmer-v5 \
    experiment.n_envs=50 \
    experiment.total_timesteps=3_000_000 \
    experiment.dir_name=results \
    experiment.render=false \
    experiment.learning_rate=1e-3 \
    experiment.n_steps=1000 \
    experiment.gamma=1 \
    experiment.seed=2026 \
    experiment.device=cpu \
    experiment.algo.sigma=0.1 \
    experiment.algo.mode=trajectory \
    experiment.algo.sampling_mode=normal \
    experiment.algo.sampling_strategy=step
