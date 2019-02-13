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
    '''Docstring'''
    frame = env.reset()
    frame = pre_process(frame)
    history_stack = deque(maxlen = 4)
    for i in range(4):
        history_stack.append(frame)
    #new_state = np.stack((frame, frame, frame, frame), axis=-1)
    is_done = False
    while not is_done:
        #counter += 1
        action  = agent.get_action(frame)
        new_frame, reward, is_done, _ = env.step(action)
        new_frame = pre_process(new_frame)
        new_state[-1] = new_frame
        agent.add_experience(state, action, reward, new_state, done)
        if len(agent.memory) >= agent.batch_size:
            experience_batch = agent.sample_experience()
            agent.network.train(experience_batch, agent.target_network)

        frame = new_frame

env = gym.make('MsPacman-v0')
#counter = 0
def train(num_episodes):
    '''Docstring'''
    frame = env.reset()
    frame = pre_process(frame)
    state = np.stack((frame, frame, frame, frame), axis=-1)
    state = frame
    state_size = state.shape
    print(state_size)
    action_size = env.action_space.n
    DQNAgent = DQN_Agent(state_size, action_size,
                         batch_size = 32,
                         discount_factor = 0.95,
                         learning_rate = 0.00025,
                         epsilon = 0.1)

    for eps in range(num_episodes):
        episode(DQNAgent)


if __name__ == "__main__":
    num_episodes = 1
    train(num_episodes)
