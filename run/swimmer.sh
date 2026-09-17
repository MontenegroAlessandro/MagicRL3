python3 run.py -m \
    wandb.project=det-pg \
    algo@experiment.algo=fdpg \
    experiment.env_name=HalfCheetah-v5 \
    experiment.n_envs=1 \
    experiment.total_timesteps=5_000_000 \
    experiment.dir_name=results \
    experiment.render=false \
    experiment.learning_rate=1e-4 \
    experiment.n_steps=100 \
    experiment.gamma=1 \
    experiment.seed=1,2,3,4,5 \
    experiment.device=cpu \
    experiment.algo.sigma=0.5 \
    experiment.algo.mode=step \
    experiment.algo.sampling_mode=normal \
    experiment.algo.sampling_strategy=step 

# python3 run.py -m \
#     wandb.project=det-pg \
#     algo@experiment.algo=gpomdp \
#     experiment.env_name=HalfCheetah-v5 \
#     experiment.n_envs=100 \
#     experiment.total_timesteps=5_000_000 \
#     experiment.dir_name=results \
#     experiment.render=false \
#     experiment.learning_rate=1e-3 \
#     experiment.n_steps=100 \
#     experiment.gamma=1 \
#     experiment.seed=1,2,3,4,5 \
#     experiment.device=cpu \
#     experiment.algo.sigma=0.1 
