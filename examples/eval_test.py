import gymnasium as gym
from stable_baselines3 import PPO


def main():
    # load model
    model = PPO.load("results/ppo_halfcheetah")

    # create evaluation environment
    eval_env = gym.make("HalfCheetah-v4", render_mode="human")

    obs, info = eval_env.reset()

    for i in range(1000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)

        if terminated or truncated:
            obs, info = eval_env.reset()

    eval_env.close()


if __name__ == "__main__":
    main()