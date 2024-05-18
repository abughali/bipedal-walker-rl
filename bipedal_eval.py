import warnings

# Suppress AVX2 warning
warnings.filterwarnings("ignore", message="Your system is avx2 capable but pygame was not built with support for it")

import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

environment_name = "BipedalWalker-v3"

env = make_vec_env(environment_name, env_kwargs={"hardcore": False, "render_mode": "human"})

# Load the trained model

model_path = os.path.join('training', 'saved_models', 'best_model')
model = PPO.load(model_path, env=env)

# Evaluate the trained model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5, render=True)
print(f"Mean reward: {mean_reward} ± {std_reward}")

env.close()
