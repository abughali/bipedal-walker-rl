import os
import warnings
import gymnasium as gym
import numpy as np
import wandb
from stable_baselines3.ddpg import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, StopTrainingOnNoModelImprovement

# Suppress AVX2 warning
warnings.filterwarnings("ignore", message="Your system is avx2 capable but pygame was not built with support for it")

# Environment name
environment_name = "BipedalWalker-v3"

# Tuned hyperparameters
config = {
    'policy': 'MlpPolicy',
    'buffer_size': 200000,
    'batch_size': 256,
    'gamma': 0.98,
    'learning_rate': 0.0001,
    'learning_starts': 10000,
    'gradient_steps': -1,
    'train_freq': (1, 'episode')
}

# Initialize Weights and Biases (wandb) for experiment tracking
run = wandb.init(
    project="bipedal-walker",
    config=config,
    sync_tensorboard=True,  # Sync tensorboard
    monitor_gym=True,       # Monitor Gym
    save_code=True,         # Save code
)

def make_env():
    """Create and wrap the environment."""
    env = gym.make(environment_name)
    env = Monitor(env)
    return env

# Create the environment
env = DummyVecEnv([make_env])

# Define the policy_kwargs to specify the network architecture
policy_kwargs = dict(net_arch=[512, 256])

# Define action noise for exploration
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# Create the DDPG model
model = DDPG(
    env=env,
    verbose=1,
    tensorboard_log=f"runs/{run.id}",
    **config,
    policy_kwargs=policy_kwargs,
    action_noise=action_noise
)

# Callback to stop training once reward threshold is reached
callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=300, verbose=1)

# Evaluation callback to periodically evaluate and save the best model
eval_callback = EvalCallback(
    env,
    eval_freq=2000,
    deterministic=True,
    best_model_save_path='./logs/',
    verbose=1
)

# Start training the model
model.learn(
    total_timesteps=500_000,
    callback=[
        WandbCallback(
            gradient_save_freq=1000,
            model_save_path=f"models/{run.id}",
            verbose=2,
        ),
        eval_callback
    ]
)

# Save the trained model
DDPG_path = os.path.join('training', 'saved_models', 'DDPG_BipedalWalker_V3')
model.save(DDPG_path)

# Evaluate the trained model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Finish the wandb run
run.finish()

# Close the environment
env.close()