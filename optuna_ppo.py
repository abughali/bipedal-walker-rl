import os
from typing import Any, Dict, Optional
import gymnasium as gym
import optuna
import optuna.visualization as vis
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback, EvalCallback, StopTrainingOnNoModelImprovement, StopTrainingOnRewardThreshold
)
from stable_baselines3.common.monitor import Monitor
import torch
import torch.nn as nn
import warnings

# Suppress AVX2 warning
warnings.filterwarnings("ignore", message="Your system is avx2 capable but pygame was not built with support for it")

# Hyperparameters and environment configuration
N_TRIALS = 8
N_STARTUP_TRIALS = 5
N_EVALUATIONS = 150
N_TIMESTEPS = 150_000
EVAL_FREQ = N_TIMESTEPS // N_EVALUATIONS
N_EVAL_EPISODES = 5
N_JOBS = 8
STUDY_PATH = "optuna/study/ppo"
ENV_ID = "BipedalWalker-v3"

# Default hyperparameters
DEFAULT_HYPERPARAMS = {
    "policy": "MlpPolicy",
    "env": ENV_ID,
}

# Linear learning rate schedule
def linear_schedule(initial_value):
    def func(progress_remaining):
        return progress_remaining * initial_value
    return func

# Function to sample PPO hyperparameters
def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    gamma = trial.suggest_float('gamma', 0.9, 0.9999, log=True)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    lr_schedule = trial.suggest_categorical('lr_schedule', ['linear', 'constant'])
    ent_coef = trial.suggest_float("ent_coef", 0.0000001, 0.1, log=True)
    clip_range = trial.suggest_float('clip_range', 0.1, 0.3, log=True)
    n_epochs = trial.suggest_int('n_epochs', 1, 10)
    gae_lambda = trial.suggest_float('gae_lambda', 0.8, 1.0, log=True)
    max_grad_norm = trial.suggest_float('max_grad_norm', 0.3, 5.0, log=True)
    vf_coef = trial.suggest_float("vf_coef", 0.0001, 1, log=True)
    ortho_init = trial.suggest_categorical('ortho_init', [False, True])
    activation_fn = trial.suggest_categorical('activation_fn', ['tanh', 'relu'])
    normalize_advantage = trial.suggest_categorical("normalize_advantage", [False, True])

    # Suppress truncated mini-batch warning
    if batch_size > n_steps:
        batch_size = n_steps
        trial.set_user_attr("batch_size_", batch_size)

    if lr_schedule == "linear":
        learning_rate = linear_schedule(learning_rate)

    # Map net_arch to architecture
    net_arch_width = trial.suggest_categorical("net_arch_width", [64, 128, 256, 512])
    net_arch_depth = trial.suggest_int("net_arch_depth", 1, 3)
    net_arch = dict(pi=[net_arch_width] * net_arch_depth, vf=[net_arch_width] * net_arch_depth)

    # Map activation_fn to torch functions
    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation_fn]

    return {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "normalize_advantage": normalize_advantage,
        "ent_coef": ent_coef,
        "clip_range": clip_range,
        "n_epochs": n_epochs,
        "gae_lambda": gae_lambda,
        "max_grad_norm": max_grad_norm,
        "vf_coef": vf_coef,
        "policy_kwargs": dict(
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
        ),
    }

# Callback used for evaluating and reporting a trial.
class TrialEvalCallback(EvalCallback):

    def __init__(
        self,
        eval_env: gym.Env,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 0,
        callback_on_new_best: Optional[BaseCallback] = None,
        callback_after_eval: Optional[BaseCallback] = None
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
            callback_on_new_best=callback_on_new_best,
            callback_after_eval=callback_after_eval
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if need.
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True

# Define the objective function for Optuna
def objective(trial: optuna.Trial) -> float:
    kwargs = DEFAULT_HYPERPARAMS.copy()
    # Sample hyperparameters.
    kwargs.update(sample_ppo_params(trial))
    # Create the RL model.
    model = PPO(**kwargs)
    # Create env used for evaluation.
    eval_env = Monitor(gym.make(ENV_ID))
    # Create the callback that will periodically evaluate and report the performance.

    stop_max_reward = StopTrainingOnRewardThreshold(reward_threshold=300, verbose=1)
    stop_no_improve = StopTrainingOnNoModelImprovement(max_no_improvement_evals=30, min_evals=50, verbose=1)
    
    eval_callback = TrialEvalCallback(eval_env, 
                                      trial,
                                      n_eval_episodes=N_EVAL_EPISODES, 
                                      eval_freq=EVAL_FREQ, 
                                      deterministic=True,
                                      verbose=1,
                                      #callback_on_new_best=stop_max_reward,
                                      #callback_after_eval=stop_no_improve
    )

    nan_encountered = False
    try:
        # Train the model with pruning
        model.learn(N_TIMESTEPS, callback=eval_callback)
    except (AssertionError, ValueError) as e:
        # Sometimes, random hyperparams can generate NaN.
        print(e)
        nan_encountered = True
    finally:
        # Free memory.
        model.env.close()
        eval_env.close()

    # Tell the optimizer that the trial failed.
    if nan_encountered:
        return float("nan")

    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    return eval_callback.last_mean_reward

# Optimize hyperparameters using Optuna
if __name__ == "__main__":
    # Control pytorch num threads for faster training.
    torch.set_num_threads(N_JOBS)

    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    # Do not prune before 1/3 of the max budget is used.
    pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS, 
                          n_warmup_steps=N_EVALUATIONS // 3)
    
    storage = 'sqlite:///example.db'

    # Make Optuna study re-runnable
    study = optuna.create_study(storage=storage,
                                study_name='bipedal_v3_normal_ppo',
                                sampler=sampler, 
                                pruner=pruner, 
                                load_if_exists=True,
                                direction="maximize")
    # continue if interrupted
    try:
        study.optimize(objective, 
                       n_jobs=N_JOBS,
                       n_trials=N_TRIALS)
    except KeyboardInterrupt:
        pass

    print("Number of finished trials: ", len(study.trials))

    os.makedirs(STUDY_PATH, exist_ok=True)

    # In addition to embedded sql-lite db, save in csv
    study.trials_dataframe().to_csv(f"{STUDY_PATH}/report.csv")

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print("    {}: {}".format(key, value))

    try:

        fig1 = vis.plot_optimization_history(study)
        fig2 = vis.plot_param_importances(study)
        fig3 = vis.plot_parallel_coordinate(study)

        # 3 browser windows will be open
        fig1.show()
        fig2.show()
        fig3.show()

    except (ValueError, ImportError, RuntimeError) as e:
        print("Error during plotting")
        print(e)