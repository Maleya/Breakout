import keras
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense
import gym
import numpy as np
import random as rnd

from collections import deque

from NN_ver1 import DQN_net
from pre_process import pre_process
from DQN_agent_ver1 import DQN_Agent

def episode(agent):
    frame = env.reset()
    frame = pre_process(frame)
    is_done = False
    while not is_done:
      # Perform a random action, returns the new frame, reward and whether the game is over
      new_frame, reward, is_done, _ = env.step(action)
      new_frame = pre_process(new_frame)




def train(num_episodes):
    env = gym.make('MsPacman-v0')
    frame = env.reset()
    frame = pre_process(frame)
    state = np.stack((frame, frame, frame, frame), axis=-1)
    state_size = state.shape
    print(state_size)
    action_size = env.action_space.n
    DQNAgent = DQN_Agent(state_size, action_size)

    for episode in range(num_episodes):
        pass


if __name__ == "__main__":
    num_episodes = 1
    train(num_episodes)
