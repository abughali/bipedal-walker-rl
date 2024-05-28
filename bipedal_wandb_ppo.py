import warnings

# Suppress AVX2 warning
warnings.filterwarnings("ignore", message="Your system is avx2 capable but pygame was not built with support for it")

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

environment_name = "BipedalWalker-v3"




# tuned hyperparamters
config = {
    'policy': 'MlpPolicy',
    'n_steps': 1503, 
    'batch_size': 32, 
    'gamma': 0.9606344595370512, 
    'learning_rate': 0.0004803031415590949, 
    'ent_coef': 5.2492091292781885e-06, 
    'clip_range': 0.19171786835210225, 
    'n_epochs': 2, 
    'gae_lambda': 0.8571941809479062, 
    'max_grad_norm': 2.372533951435877, 
    'vf_coef': 0.39445800472492965
}

run = wandb.init(
    project="bipedal-walker",
    config=config,
    sync_tensorboard=True,  
    monitor_gym=True,  
    save_code=True,  
)


# Create and wrap the environment
def make_env():
    env = gym.make(environment_name)
    env = Monitor(env)
    return env

env = DummyVecEnv([make_env])

#env = VecVideoRecorder(env, f"videos/{run.id}", record_video_trigger=lambda x: x % 200 == 0, video_length=1000)

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
callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=310, verbose=1)


# Use deterministic actions for evaluation
eval_callback = EvalCallback(env, 
                            # callback_on_new_best=callback_on_best, 
                             eval_freq=2000,
                             deterministic=False, 
                             best_model_save_path='./logs/', 
                             verbose=1)

model.learn(
            total_timesteps=500_000,
            callback=[WandbCallback(
                        gradient_save_freq=1000,
                        model_save_path=f"models/{run.id}",
                        verbose=2,
                    ), eval_callback]
        )


PPO_path = os.path.join('training', 'saved_models', 'PPO_BipedalWalker_Normal_Det')
model.save(PPO_path)

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

run.finish()
env.close()