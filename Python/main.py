"""
info
"""

import gym
import numpy as np
import random as rnd
from collections import deque
from Neuralnet import Neuralnet
from pre_process import pre_process
from Agent import Agent
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

env = gym.make('MsPacman-v0')


def episode(learning_Agent):
    """ An episode is one game session,    """
    assert type(learning_Agent) == Agent

    points = 0
    frame = env.reset()
    frame = pre_process(frame)
    history_stack= deque([frame for i in range(4)], maxlen=4) # is this correct?
    state = np.stack((frame,)*4, axis=-1)
    state = np.expand_dims(state, axis=0)

    is_done = False
    while not is_done:
        env.render()
        action = learning_Agent.get_action(state)
        new_frame, reward, is_done, _ = env.step(action)
        points += reward

        # ..
        new_frame = pre_process(new_frame)
        history_stack.append(new_frame)  # add latest frame
        new_state = np.stack((elem for elem in history_stack),axis=-1)
        new_state = np.expand_dims(new_state, axis=0)
        learning_Agent.add_experience(state, action, reward, new_state, is_done)

        # update network weights: (and reset them when needed)
        if len(learning_Agent.memory) >= learning_Agent.batch_size*2:
            experience_batch = learning_Agent.sample_experience()
            learning_Agent.network.train(experience_batch, learning_Agent.target_network)
            learning_Agent.learning_count += 1
            if learning_Agent.learning_count % learning_Agent.learning_count_max == 0:
                learning_Agent.reset_target_network()

        state = new_state

    return points


def training(num_episodes):
    '''Docstring'''
    # we generat the right sizes for frame and actionspace
    frame = env.reset()
    frame = pre_process(frame)
    state_size = np.stack((frame,)*4, axis=-1).shape
    print('state size =', state_size)
    action_size = env.action_space.n

    DQNAgent = Agent(state_size, action_size,
                         batch_size = 32,
                         discount_factor = 0.95,
                         learning_rate = 0.00025,
                         epsilon = 0.2)
    points_history = []
    for eps in range(num_episodes):
        points = episode(DQNAgent)
        points_history.append(points)
        print('points for episode:', eps, points)
    return points_history


if __name__ == "__main__":
    num_episodes = 1000
    Return_history = training(num_episodes)
    episodes_v = [i for i in range(num_episodes)]
    plt.plot(episodes_v, Return_history, '.')
    plt.savefig('Boxing_score_vr_episodes_#1000.pdf')
    plt.show()
