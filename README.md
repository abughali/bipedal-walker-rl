# Bipedal Walker

This is how to teach bot to walk on two feet.

Bipedal Walker is an [OpenAI Gym](https://gymnasium.farama.org/environments/box2d/bipedal_walker/) environment where an agent learns to control a bipedal 2D character to reach the end of an obstacle course. What makes this challenging is that the agent only receives limbs coordinates along with the Lidar information. The agent has to learn to balance, walk, run, jump on its own without any human intervention.

## Prerequisites and Recommended Tools

- [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) installed on your system
- Visual Studio Code

## Setup Instructions

Follow the steps below to set up your environment and install the necessary dependencies.

##### 1. Create a Conda Environment

First, create a new Conda environment with Python 3.11:

```sh
conda create -n gym-env python=3.11 -y
```

##### 2.  Activate the newly created environment
```sh
conda activate gym-env
```

##### 3.  Install the required packages
```sh
conda install swig gymnasium-box2d stable-baselines3 tensorboard -y
```

##### 4. Verify the Installation
To ensure that everything is set up correctly, run a quick test to check the installed packages:
```sh
python -c "import gymnasium; import stable_baselines3; import tensorboard; print('All packages installed successfully!')"
```

##### 5. Clone this github repository.

Once the environment is set up and dependencies are installed, you can proceed to train the agent and evaluate its performance.

### Project Structure
- `bipedal_train.py`: Script to train the PPO agent.
- `bipedal_eval.py`: Script to evaluate the trained PPO agent.

### Tensorboard Visualization

```sh
tensorboard --logdir=logs
```

### Additional Information
- [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/en/master/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [TensorBoard Documentation](https://www.tensorflow.org/tensorboard/get_started)

### License

This project is licensed under the MIT License.