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


env = gym.make('MsPacman-v0')

def episode(agent):
    '''Docstring'''
    frame = env.reset()
    frame = pre_process(frame)
    history_stack = deque(maxlen = 4)

    for i in range(4):
        history_stack.append(frame)
    state = np.stack((frame, frame, frame, frame), axis=-1)
    state = np.expand_dims(state, axis=0)
    is_done = False
    while not is_done:
        #counter += 1
        action  = agent.get_action(state)
        new_frame, reward, is_done, _ = env.step(action)
        #Procecessing images to 4D tensor for the conv_2D input
        new_frame = pre_process(new_frame)
        history_stack.append(new_frame)
        new_state = np.stack((history_stack[0], history_stack[1], history_stack[2], history_stack[3]), axis=-1)
        new_state = np.expand_dims(new_state, axis=0)
        agent.add_experience(state, action, reward, new_state, is_done)
        '''
        if len(agent.memory) >= agent.batch_size:
            experience_batch = agent.sample_experience()
            agent.network.train(experience_batch, agent.target_network)
            '''
        state = new_state
    if len(agent.memory) >= agent.batch_size:
        experience_batch = agent.sample_experience()
        agent.network.train(experience_batch, agent.target_network)
#counter = 0
def train(num_episodes):
    '''Docstring'''
    frame = env.reset()
    frame = pre_process(frame)
    state = np.stack((frame, frame, frame, frame), axis=-1)
    #state = frame
    state_size = state.shape
    print('state size =',state_size)
    action_size = env.action_space.n
    DQNAgent = DQN_Agent(state_size, action_size,
                         batch_size = 32,
                         discount_factor = 0.95,
                         learning_rate = 0.00025,
                         epsilon = 0)

    for eps in range(num_episodes):
        episode(DQNAgent)


if __name__ == "__main__":
    num_episodes = 1
    train(num_episodes)
