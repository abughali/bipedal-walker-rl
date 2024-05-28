import os
import gymnasium as gym
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3 import PPO, DDPG
import warnings

# Suppress AVX2 warning
warnings.filterwarnings("ignore", message="Your system is avx2 capable but pygame was not built with support for it")

def record_video(env_id, model, video_length=1600, prefix="", video_folder="videos/"):
    """
    Record a video of the model interacting with the environment.

    :param env_id: (str) The environment ID
    :param model: (RL model) The trained RL model
    :param video_length: (int) The number of timesteps to record
    :param prefix: (str) Prefix for the video file name
    :param video_folder: (str) Directory to save the video
    """
    eval_env = DummyVecEnv([lambda: gym.make(env_id, render_mode="rgb_array")])
    # Start the video at step=0 and record `video_length` steps
    eval_env = VecVideoRecorder(
        eval_env,
        video_folder=video_folder,
        record_video_trigger=lambda step: step == 0,
        video_length=video_length,
        name_prefix=prefix,
    )

    obs = eval_env.reset()
    total_reward = 0
    timesteps = 0
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, done, _ = eval_env.step(action)
        total_reward += rewards
        timesteps += 1

    # Print the total reward and total timesteps
    print(f"Total reward for the recorded episode: {total_reward}, Total timesteps: {timesteps}")

    # Close the video recorder
    eval_env.close()

if __name__ == "__main__":
    environment_name = "BipedalWalker-v3"
    ALGO = PPO  # or DDPG

    # Load the trained model
    model_path = os.path.join('training', 'saved_models', 'PPO_BipedalWalker_V3')
    model = ALGO.load(model_path)

    # Record the video
    record_video(environment_name, model, video_length=1600, prefix="final_bipedalwalker", video_folder="videos/ppo")
