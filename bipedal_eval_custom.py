import warnings

# Suppress AVX2 warning
warnings.filterwarnings("ignore", message="Your system is avx2 capable but pygame was not built with support for it")

import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import gym

environment_name = "BipedalWalker-v3"

gym.envs.register(
     id='NewEnvBipedal-v4',
     entry_point='custom_env.bipedal_walker:BipedalWalker',
     max_episode_steps=1000,
     reward_threshold=300.0
)

def make_env():
    return gym.make("NewEnvBipedal-v4", render_mode="human")

env = make_env()

env = make_vec_env(lambda: env)

# Load the trained model

model_path = os.path.join('training', 'saved_models', 'PPO_BipedalWalker_final_custom')
model = PPO.load(model_path, env=env)

# Evaluate the trained model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1, render=True)
print(f"Mean reward: {mean_reward} ± {std_reward}")

env.close()
