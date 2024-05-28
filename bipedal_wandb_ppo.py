import warnings
import os
import wandb
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, StopTrainingOnNoModelImprovement
import torch as th

# Suppress AVX2 warning
warnings.filterwarnings("ignore", message="Your system is avx2 capable but pygame was not built with support for it")

# Environment name
environment_name = "BipedalWalker-v3"

# Tuned hyperparameters
config = {
    'policy': 'MlpPolicy',
    'n_steps': 2048, 
    'batch_size': 32, 
    'gamma': 0.96, 
    'learning_rate': 0.00048, 
    'ent_coef': 0.0000004, 
    'clip_range': 0.191, 
    'n_epochs': 3, 
    'gae_lambda': 0.857, 
    'max_grad_norm': 2.372, 
    'vf_coef': 0.394
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

env = DummyVecEnv([make_env])

# Define the policy_kwargs to specify the network architecture
policy_kwargs = {
    "net_arch": {
        "pi": [256, 256],
        "vf": [256, 256],
    },
    "activation_fn": th.nn.ReLU
}

# Create the PPO model
model = PPO(
    env=env, 
    verbose=1, 
    tensorboard_log=f"runs/{run.id}",
    policy_kwargs=policy_kwargs,
    **config
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
PPO_path = os.path.join('training', 'saved_models', 'PPO_BipedalWalker_V3')
model.save(PPO_path)

# Evaluate the trained model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

# Finish the wandb run
run.finish()

# Close the environment
env.close()
