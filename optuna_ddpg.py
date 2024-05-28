import os
import warnings
from typing import Optional

import gymnasium as gym
import numpy as np
import optuna
import optuna.visualization as vis
import torch
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from stable_baselines3.ddpg import DDPG
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

# Suppress AVX2 warning
warnings.filterwarnings("ignore", message="Your system is avx2 capable but pygame was not built with support for it")

# Hyperparameters and environment configuration
N_TRIALS = 100
N_STARTUP_TRIALS = 5
N_EVALUATIONS = 150
N_TIMESTEPS = 150_000
EVAL_FREQ = N_TIMESTEPS // N_EVALUATIONS
N_EVAL_EPISODES = 5
N_JOBS = 8
STUDY_PATH = "optuna/study/ddpg"
ENV_ID = "BipedalWalker-v3"

# Default hyperparameters
DEFAULT_HYPERPARAMS = {
    "policy": "MlpPolicy",
    "env": ENV_ID,
}

#Linear learning rate schedule.
def linear_schedule(initial_value):
    def func(progress_remaining):
        return progress_remaining * initial_value
    return func

#Function to sample DDPG hyperparameters.
def sample_ddpg_params(trial, n_actions):

    gamma = trial.suggest_float('gamma', 0.9, 0.9999, log=True)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 100, 128, 256, 512])
    buffer_size = trial.suggest_categorical("buffer_size", [int(1e4), int(1e5), int(1e6)])
    tau = trial.suggest_categorical("tau", [0.001, 0.005, 0.01, 0.02, 0.05, 0.08])
    train_freq = trial.suggest_categorical("train_freq", [1, 4, 8, 16, 32, 64, 128])
    gradient_steps = train_freq
    noise_type = trial.suggest_categorical("noise_type", ["ornstein-uhlenbeck", "normal", None])
    noise_std = trial.suggest_float("noise_std", 0, 1)
    net_arch_type = trial.suggest_categorical("net_arch", ["small", "medium", "big"])
    net_arch = {"small": [64, 64], "medium": [256, 256], "big": [512, 512]}[net_arch_type]

    hyperparams = {
        "gamma": gamma,
        "tau": tau,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
        "policy_kwargs": dict(net_arch=net_arch),
    }

    if noise_type == "normal":
        hyperparams["action_noise"] = NormalActionNoise(
            mean=np.zeros(n_actions), sigma=noise_std * np.ones(n_actions)
        )
    elif noise_type == "ornstein-uhlenbeck":
        hyperparams["action_noise"] = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(n_actions), sigma=noise_std * np.ones(n_actions)
        )

    return hyperparams

#Callback used for evaluating and reporting a trial.
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
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True

#Define the objective function for Optuna.

def objective(trial: optuna.Trial) -> float:

    kwargs = DEFAULT_HYPERPARAMS.copy()
    eval_env = Monitor(gym.make(ENV_ID))
    n_actions = eval_env.action_space.shape[0]

    # Sample hyperparameters.
    kwargs.update(sample_ddpg_params(trial, n_actions))
    # Create the RL model.
    model = DDPG(**kwargs)
    
    # Create the callback that will periodically evaluate and report the performance.
    eval_callback = TrialEvalCallback(eval_env, 
                                      trial,
                                      n_eval_episodes=N_EVAL_EPISODES, 
                                      eval_freq=EVAL_FREQ, 
                                      deterministic=True,
                                      verbose=1
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
                                study_name='bipedal_v3_normal_ddpg',
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
