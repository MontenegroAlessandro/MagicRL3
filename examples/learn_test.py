import gymnasium as gym
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.env_util import make_vec_env
import wandb
from wandb.integration.sb3 import WandbCallback

# parameters
config = dict(
    policy_type="MlpPolicy",
    total_timesteps=3_000_000,
    env_id="HalfCheetah-v5",
    n_envs=6,
    dir_name="results",
)

# logger 
run = wandb.init(
    project="sb3-ppo", 
    config=config, 
    sync_tensorboard=True
)

# make the env
env = make_vec_env(config["env_id"], n_envs=config["n_envs"], seed=2026)

# learn
model = A2C(
    config["policy_type"], 
    env, 
    verbose=0, 
    tensorboard_log=f"{config['dir_name']}/runs/{run.id}",
    seed=2026
)
model.learn(
    total_timesteps=config["total_timesteps"], 
    progress_bar=True,
    callback=WandbCallback(
        model_save_path=f"{config['dir_name']}/models/{run.id}", 
        verbose=2
    )
)
model.save(f"{config['dir_name']}/ppo_halfcheetah")

# evaluate
eval_env = gym.make(config["env_id"], render_mode="human")
obs, info = eval_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = eval_env.step(action)
    if terminated or truncated:
        obs, info = eval_env.reset()
eval_env.close()