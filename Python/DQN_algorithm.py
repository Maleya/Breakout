#import keras
#from keras.models import Sequential
#from keras.layers import Conv2D, Flatten, Dense
import gym
import numpy as np
import random as rnd
from collections import deque
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
# Our documents
from NN_ver1 import DQN_net
from pre_process import pre_process
from DQN_agent_ver1 import DQN_Agent
from stack_frames import stack_frames
from preprocess_BO import pre_process_BO
env = gym.make('Breakout-v0')

def episode(agent):
    '''Docstring'''
    Return = 0
    initial_frame = env.reset()
    #initial_frame = pre_process(initial_frame) #MsPacman
    initial_frame = pre_process_BO(initial_frame) #Breakout
    stacked_frames = stack_frames()
    state = stacked_frames.create_stack(initial_frame)
    counter = 0
    is_done = False
    while not is_done:
        if agent.video == True:
            env.render()

        counter += 1
        action  = agent.get_action(state)
        new_frame, reward, is_done, _ = env.step(action)
        Return += reward
        #Procecessing images to 4D tensor for the conv_2D input
        #new_frame = pre_process(new_frame) # MsPacman
        new_frame = pre_process_BO(new_frame) #Breakout
        new_state = stacked_frames.get_new_state(new_frame)

        agent.add_experience(state, action, reward, new_state, is_done)

        #Network wights update:
        if len(agent.memory) >= agent.batch_size*2:
            experience_batch = agent.sample_experience()
            agent.network.train(experience_batch, agent.target_network)
            agent.learning_count += 1
            if agent.learning_count % agent.learning_count_max == 0:
                agent.reset_target_network()
        state = new_state
    if agent.video == True:
    print(counter)
    return Return
#counter = 0
def train(num_episodes):
    '''Docstring'''
    frame = env.reset()
    frame = pre_process_BO(frame)
    state = np.stack((frame, frame, frame, frame), axis=-1)
    #state = frame
    state_size = state.shape
    print('state size =',state_size)
    action_size = env.action_space.n
    print('action size =', action_size)
    DQNAgent = DQN_Agent(state_size, action_size,
                         batch_size = 32,
                         discount_factor = 0.95,
                         learning_rate = 0.00025,
                         epsilon = 0.2,
                         video = True)
    Return_history = []
    for eps in range(num_episodes):
        Return = episode(DQNAgent)
        Return_history.append(Return)
        print('Return for episode:',eps,'is:',Return)

    return Return_history


if __name__ == "__main__":
    num_episodes = 10
    Return_history = train(num_episodes)
    episodes_v = [i for i in range(num_episodes)]
    env.close()
    plt.plot(episodes_v, Return_history, '.')
    plt.savefig('Breakout_score_vr_episodes_#500.pdf')
    plt.show()
