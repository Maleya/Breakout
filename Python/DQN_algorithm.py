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
        if len(agent.memory) >= agent.batch_size*500:
            experience_batch = agent.sample_experience()
            agent.network.train(experience_batch, agent.target_network)
            agent.learning_count += 1
            if agent.learning_count % agent.learning_count_max == 0:
                agent.reset_target_network()

            if agent.epsilon > agent.min_epsilon:
                agent.epsilon *= agent.epsilon_decay
        state = new_state

    #----PRINTS FOR TESTING ----------------------
    print(f'number of steps in episode = {counter}')
    print(f'epsilon = {agent.epsilon}')

    #---------------------------------------------
    return Return


def training(num_learning_episodes):
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
                         epsilon=1,
                         epsilon_decrease_rate=0.9999,
                         min_epsilon=0.1,
                         video = False,
                         epsilon_linear = True)
    mean_history = []
    Return_history = []
    #for eps in range(num_episodes):
    episode_count = 0
    while DQNAgent.learning_count < num_learning_episodes:
        episode_count += 1
        Return = episode(DQNAgent)
        mean_history.append(Return)
        if episode_count%100 == 0:
            Return_history.append(np.mean(mean_history))
        print('Return for episode:',episode_count,'is:',Return)

    return Return_history


if __name__ == "__main__":
    num_episodes = 1000
    num_learning_episodes = 1000
    Return_history = training(num_learning_episodes)
    episodes_v = [i for i in range(int(len(Return_history)))]
    env.close()
    plt.plot(episodes_v, Return_history, '.')
    plt.savefig('Breakout_score_vr_epochs_#100.pdf')
    plt.show()
