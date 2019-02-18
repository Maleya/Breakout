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
        if len(agent.memory) >= learning_Agent.batch_size*2:
            experience_batch = learning_Agent.sample_experience()
            learning_Agent.network.train(experience_batch, learning_Agent.target_network)
            learning_Agent.learning_count += 1
            if learning_Agent.learning_count % learning_Agent.learning_count_max == 0:
                learning_Agent.reset_target_network()
       
        state = new_state

    return points


