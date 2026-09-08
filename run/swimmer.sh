# python3 run.py -m \
#     wandb.project=det-pg \
#     algo@experiment.algo=fdpg \
#     experiment.env_name=Swimmer-v5 \
#     experiment.n_envs=5 \
#     experiment.total_timesteps=5_000_000 \
#     experiment.dir_name=results \
#     experiment.render=false \
#     experiment.learning_rate=1e-3 \
#     experiment.n_steps=200 \
#     experiment.gamma=1 \
#     experiment.seed=2,3,4,5 \
#     experiment.device=cpu \
#     experiment.algo.sigma=0.1 \
#     experiment.algo.mode=trajectory \
#     experiment.algo.sampling_mode=normal \

python3 run.py -m \
    wandb.project=det-pg \
    algo@experiment.algo=reinforce,gpomdp \
    experiment.env_name=Swimmer-v5 \
    experiment.n_envs=10 \
    experiment.total_timesteps=5_000_000 \
    experiment.dir_name=results \
    experiment.render=false \
    experiment.learning_rate=1e-3 \
    experiment.n_steps=200 \
    experiment.gamma=1 \
    experiment.seed=1,2,3,4,5 \
    experiment.device=cpu \
    experiment.algo.sigma=0.1 
