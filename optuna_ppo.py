import gymnasium as gym
import optuna
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import optuna.visualization as vis


# Function to sample hyperparameters
def sample_ppo_params(trial):

    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512])
    n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])

    if batch_size > n_steps:
       batch_size = n_steps

    return {
        'n_steps': n_steps,
        'batch_size': batch_size,
        'gamma': trial.suggest_float('gamma', 0.9, 0.9999, log=True),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'ent_coef': trial.suggest_float('ent_coef', 0.00000001, 0.1, log=True),
        'clip_range': trial.suggest_float('clip_range', 0.1, 0.4, log=True),
        'n_epochs': trial.suggest_int('n_epochs', 1, 10),
        'gae_lambda': trial.suggest_float('gae_lambda', 0.8, 1.0, log=True),
        'max_grad_norm': trial.suggest_float('max_grad_norm', 0.3, 5.0, log=True),
        'vf_coef': trial.suggest_float('vf_coef', 0.1, 1.0, log=True),
        'net_arch': trial.suggest_categorical('net_arch', ['small', 'medium', 'large']),
        'activation_fn': trial.suggest_categorical('activation_fn', ['tanh', 'relu'])
    }

# Map net_arch to architecture
net_arch = {
    'small': dict(pi=[64, 64], vf=[64, 64]),
    'medium': dict(pi=[128, 128], vf=[128, 128]),
    'large': dict(pi=[256, 256], vf=[256, 256])
}

# Map activation_fn to torch functions
activation_fn = {
    'tanh': th.nn.Tanh,
    'relu': th.nn.ReLU
}

# Define the objective function for Optuna
def objective(trial):
    env_id = "BipedalWalker-v3"
    env = make_vec_env(env_id, seed=0, n_envs=1)
    
    # Sample hyperparameters
    hyperparams = sample_ppo_params(trial)
    
    # Convert net_arch and activation_fn to the correct format
    hyperparams['policy_kwargs'] = {
        'net_arch': net_arch[hyperparams.pop('net_arch')],
        'activation_fn': activation_fn[hyperparams.pop('activation_fn')]
    }
    
    # Create the PPO model
    model = PPO('MlpPolicy', env, verbose=0, **hyperparams)
    
    # Create an evaluation callback
    eval_env = make_vec_env(env_id, seed=1, n_envs=1)
    eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
                                 log_path='./logs/', eval_freq=1000, 
                                 deterministic=True, render=False)
    
    # Train the model with pruning
    model.learn(total_timesteps=130000, callback=[eval_callback])
    
    # Evaluate the model
    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=10)
    
    trial.report(mean_reward, step=1)
    
    # Prune trial if necessary
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()
    
    return mean_reward

# Optimize hyperparameters using Optuna
def optimize_ppo():
    sampler = TPESampler()
    pruner = MedianPruner()
    study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=100, n_jobs=1)
    
    print("Best hyperparameters: ", study.best_params)
    print("Best trial: ", study.best_trial)
    
    # Visualize the results
    fig1 = vis.plot_optimization_history(study)
    fig2 = vis.plot_param_importances(study)
    fig3 = vis.plot_parallel_coordinate(study)
    
    fig1.show()
    fig2.show()
    fig3.show()

if __name__ == '__main__':
    optimize_ppo()
