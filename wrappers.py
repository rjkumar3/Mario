import numpy as np
from gym import Wrapper
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack


# Class to combine all manipulation of acquired data from the environment
# This will Skip 4 frames for each state, reduce the size of captured images,
# remove the color of captured images, and stack consecutive frames to perceive movement

# Here we override the constructor to input out own number of fames to skip
class SkipFrame(Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self.skip = skip

# Here we override the skip function to allow us to process the rewards from the skipped frames
    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self.skip):
            next_state, reward, done, trunc, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return next_state, total_reward, done, trunc, info


def apply_wrappers(env):
    env = SkipFrame(env, skip=4)  # Num of frames to apply one action to
    env = ResizeObservation(env, shape=84)  # Resize frame from 240x256 to 84x84
    env = GrayScaleObservation(env)  # Remove color from state
    env = FrameStack(env, num_stack=4, lz4_compress=True)
    return env
