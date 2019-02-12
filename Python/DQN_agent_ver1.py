import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model, Sequential
import gym
import numpy as np
import random as rnd

from NN_ver1 import DQN_net


class DQN_Agent:
    def __init__(self, state_size, action_size,
                 batch_size=32,
                 discount_factor = 0.95,
                 learning_rate=0.00025,
                 epsilon = 0.1):

        self.state_size = state_size
        self.action_size = action_size # should be 4 for pacman
        self.memory = deque(maxlen=2000) #Replay Memory
        self.gamma = 0.95    # discount rate
        self.epsilon = 0.1 # exploration rate
        self.discount_factor = discount_factor  # Discount factor of the MDP
        self.learning_rate = learning_rate  # Learning rate
        self.epsilon = epsilon  # Probability of dropout
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.network = DQN_net(self, self.state_size, self.action_size,
                     batch_size = self.batch_size,
                     discount_factor = self.discount_factor,
                     learning_rate=self.learning_rate,
                     discount_factor=self.discount_factor,
                     epsilon=self.epsilon)

        self.target_network = DQN_net(self, self.state_size, self.action_size,
                     batch_size = self.batch_size,
                     discount_factor = self.discount_factor,
                     learning_rate=self.learning_rate,
                     discount_factor=self.discount_factor,
                     epsilon=self.epsilon)

    def add_experience(self, state, action, reward, next_state, done):
        '''If len(memory) = maxlen it will pop the oldest data from left
        and add the new data at the end of the list.
        '''
        self.memory.append((state, action, reward, next_state, done))


    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            #Selects one of the possible actions randomly
            return env.action_space.sample()

        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns index corresponding to chosen action

    def sample_experience(self):
        return random.sample(self.memory, slef.batch_size)


def q_iteration(DQN_Agent):

    if len(self.memory) > self.batch_size:
