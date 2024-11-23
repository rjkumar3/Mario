import torch
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY  # import the game environment and ability to move right

from agent import Agent

from nes_py.wrappers import JoypadSpace
from wrappers import apply_wrappers

import os

from utils import *

# Create folder directly to store models in training
model_path = os.path.join("models", get_current_date_time_string())
os.makedirs(model_path, exist_ok=True)


# check if CUDA is available to allow for training on GPU
if torch.cuda.is_available():
    print("Using CUDA device:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available")

ENV_NAME = 'SuperMarioBros-1-1-v0'
SHOULD_TRAIN = False
DISPLAY = True
CKPT_SAVE_INTERVAL = 5000  # number of itterations to be executered between saving the weights of the model to a local backup this also allows for training to be paused and continued later
NUM_OF_EPISODES = 50_000  # number of episodes to loop though to fully train the agent

env = gym_super_mario_bros.make(ENV_NAME, render_mode='human' if DISPLAY else 'rgb', apply_api_compatibility=True)  # creating environment for level 1-1, implimenting the sigle life mode, and prompting for a display to appear on the screen so we can watch our agent progress
env = JoypadSpace(env, RIGHT_ONLY)  # enable the right only movement

env = apply_wrappers(env)

agent = Agent(input_dims=env.observation_space.shape, num_actions=env.action_space.n)

# if to load a trained model for more training
# user needs to populate the directory information by providing the folder within the models folder to select the model from
# and set SHOULD_TRAIN to true
if SHOULD_TRAIN:
    folder_name = ""
    ckpt_name = ""
    agent.load_model(os.path.join("models", folder_name, ckpt_name))

# else to allow for the program to run a trained model with adjustable parameters
# user needs to populate the directory information by providing the folder within the models folder to select the model from
# and set SHOULD_TRAIN to false
else:
    folder_name = ""
    ckpt_name = ""
    agent.load_model(os.path.join("models", folder_name, ckpt_name))

    agent.epsilon = 0.1
    agent.eps_min = 0.1
    agent.eps_decay = 0.0

env.reset()
next_state, reward, done, trunc, info = env.step(action=0)  # initializing tensors to be used by the neural network

# Training loop
for i in range(NUM_OF_EPISODES):    
    print("Episode:", i)
    done = False  # Flag to indicate death or reaching flag pole
    state, _ = env.reset()  # Clear state and replace it with current state on each itteration
    total_reward = 0
    while not done:
        a = agent.choose_action(state)  # prompt the agent to take an action
        new_state, reward, done, truncated, info = env.step(a)  # populate tensors from taken action
        total_reward += reward

        # Preform learning step
        if SHOULD_TRAIN:
            agent.store_in_memory(state, a, reward, new_state, done)
            agent.learn()

        # move to next state
        state = new_state

    print("Total reward:", total_reward, "Epsilon:", agent.epsilon, "Size of replay buffer:", len(agent.replay_buffer), "Learn step counter:", agent.learn_step_counter)

    # Save model at specified increments
    if SHOULD_TRAIN and (i + 1) % CKPT_SAVE_INTERVAL == 0:
        agent.save_model(os.path.join(model_path, "model_" + str(i + 1) + "_iter.pt"))

    print("Total reward:", total_reward)

env.close()