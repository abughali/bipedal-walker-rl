import os
from stable_baselines3 import PPO
from stable_baselines3.ddpg import DDPG
from stable_baselines3.common.vec_env import VecVideoRecorder
from stable_baselines3.common.env_util import make_vec_env

environment_name = "BipedalWalker-v3"
ALGO = PPO # DDPG

model_path = os.path.join('training', 'saved_models', 'PPO_BipedalWalker_Normal_Det')
model = ALGO.load(model_path)

# Create a directory for saving videos
video_folder = "videos/ppo"
os.makedirs(video_folder, exist_ok=True)

# Create the evaluation environment wrapped with VecVideoRecorder to record episode
eval_env = make_vec_env(environment_name)
eval_env = VecVideoRecorder(eval_env, 
                            video_folder, 
                            record_video_trigger=lambda x: x == 0,
                            video_length=2000,
                            name_prefix='final_bipedalwalker')

# Reset the environment and evaluate the model to record episode
obs = eval_env.reset()
done = False
total_reward = 0
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = eval_env.step(action)
    total_reward += reward

print(f"Total reward for the recorded episode: {total_reward}")

# Close the environment to save the video
eval_env.close()
