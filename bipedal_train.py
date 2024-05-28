import os
import warnings
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

# Suppress AVX2 warning
warnings.filterwarnings("ignore", message="Your system is avx2 capable but pygame was not built with support for it")

# Environment name for the 4-joint walker robot
environment_name = "BipedalWalker-v3"

# Initialize the environment
env = gym.make(environment_name, render_mode="human")

### Understanding The Environment ##############################################################

# Action Space
print(env.action_space)  # Detect type
print(env.action_space.sample())
# Box(-1.0, 1.0, (4,), float32)
# Actions are motor speed values in the [-1, 1] range for each of the
# 4 joints at both hips and knees.

#  |-----| (body or hull)
#  |-----| \ (lidar rangefinder)
#   O   O  (hip joints)
#    \   \
#     \   \
#      O  O  (knee joints)
#     /  /
#   /   /   

# Observation Space
print(env.observation_space.sample())
# Box([-low 14], [+high 14], (24,), float32)
# State consists of hull angle speed, angular velocity, horizontal speed,
# vertical speed, position of joints and joints angular speed, legs contact
# with ground, and 10 lidar rangefinder measurements. There are no coordinates
# in the state vector.

# Random actions
# Run n episode(s)
episodes = 0
for episode in range(1, episodes + 1):
    state = env.reset()
    done = False
    truncated = False
    score = 0
    
    while not done and not truncated:
        env.render()
        action = env.action_space.sample()
        next_state, reward, done, truncated, info = env.step(action)
        score += reward
    print('Episode:{} Score:{}'.format(episode, score))

# Close the environment
env.close()

### Rewards
# Reward is given for moving forward, totaling 300+ points up to the far end.
# If the robot falls, it gets -100. Applying motor torque costs a small
# amount of points. A more optimal agent will get a better score.

### TRAINING ##############################################################

# Paths for saving models and logs
save_path = os.path.join('training', 'saved_models')
log_path = os.path.join('training', 'logs')
training_log_path = os.path.join(log_path, 'PPO')

# Create multiple environments for parallel training
env = make_vec_env(environment_name, n_envs=4, env_kwargs={"hardcore": True})

# Create a callback to stop training once a reward threshold is reached
stop_callback = StopTrainingOnRewardThreshold(reward_threshold=300, verbose=1)
eval_callback = EvalCallback(
    env,
    callback_on_new_best=stop_callback,
    eval_freq=10000,
    best_model_save_path=save_path,
    verbose=1
)

# Create the PPO model
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path)

# Train the PPO model
model.learn(total_timesteps=500_000, callback=eval_callback)

# Save the trained model
model.save("ppo_bipedalwalker")