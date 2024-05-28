import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

# Suppress AVX2 warning
import warnings
warnings.filterwarnings("ignore", message="Your system is avx2 capable but pygame was not built with support for it")

# Environment name
environment_name = "BipedalWalker-v3"

# Create the environment with specific parameters
env = make_vec_env(environment_name, env_kwargs={"hardcore": False, "render_mode": "human"})

# Load the trained model
model_path = os.path.join('training', 'saved_models', 'PPO_BipedalWalker_final')
model = PPO.load(model_path, env=env)

# Evaluate the trained model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1, render=True)
print(f"Mean reward: {mean_reward} ± {std_reward}")

# Close the environment
env.close()
