"""
This runs our main algorithm, and his the highest in the code heirachy import-wise.
note: 
num_learning_iterations starts counting after we filled agent memory with agent.batch_size*[num]
"""

import gym
import numpy as np
import time
import matplotlib
matplotlib.use("TkAgg")  # for mac users
from matplotlib import pyplot as plt
from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # reduces verbosity of tensorflow ?

# Our documents
from NN_ver1 import DQN_net
from pre_process import pre_process
from DQN_agent_ver1 import DQN_Agent
from stack_frames import stack_frames
from preprocess_BO import pre_process_BO

env = gym.make('Breakout-v0')


def episode(agent):
    '''An episode constitues one normal run of a game.
    Takes an Agent as input and returns the episode points'''

    assert type(agent) == DQN_Agent
    points = 0
    episode_steps = 0
    ep_start_time = time.time()

    # PREP A FRAME
    initial_frame = env.reset()
    # initial_frame = pre_process(initial_frame) # MsPacman
    initial_frame = pre_process_BO(initial_frame)  # Breakout
    stacked_frames = stack_frames()
    state = stacked_frames.create_stack(initial_frame)
    is_done = False

    # RUN A FULL EPISODE:
    while not is_done:
        if agent.video is True:
            env.render()

        episode_steps += 1
        action = agent.get_action(state)
        new_frame, reward, is_done, _ = env.step(action)
        points += reward
        # Procecessing images to 4D tensor for the conv_2D input
        # new_frame = pre_process(new_frame) # MsPacman
        new_frame = pre_process_BO(new_frame)  # Breakout
        new_state = stacked_frames.get_new_state(new_frame)
        agent.add_experience(state, action, reward, new_state, is_done)

        # Network weights update: starts after delay.
        if len(agent.memory) >= agent.batch_size*500:  # sets the learning delay
            experience_batch = agent.sample_experience()
            agent.network.train(experience_batch, agent.target_network)
            agent.learning_count += 1
            if agent.learning_count % agent.learning_count_max == 0:
                agent.reset_target_network()

            # decaying epsilon.
            if agent.epsilon > agent.min_epsilon: 
                agent.epsilon *= agent.epsilon_decay
        state = new_state

    # ----PRINTS FOR TESTING ----------------------
    print(f'Took {episode_steps} steps in {round(time.time()-ep_start_time,3)} seconds with ε = {round(agent.epsilon,3)}')

    # ---------------------------------------------
    return points


def run_training(num_learning_iterations):
    '''Docstring'''
    frame = env.reset()
    frame = pre_process_BO(frame)
    state = np.stack((frame,)*4, axis=-1)
    mean_history = []
    points_history = []
    episode_count = 0
    state_size = state.shape
    action_size = env.action_space.n

    # print('state size =', state_size)
    # print('action size =', action_size)

    DQNAgent = DQN_Agent(state_size, action_size,
                         batch_size = 32,
                         discount_factor = 0.95,
                         learning_rate = 0.00025,
                         epsilon=1,
                         epsilon_decrease_rate=0.9999,
                         min_epsilon=0.1,
                         video = False,
                         epsilon_linear = True)

    while DQNAgent.learning_count < num_learning_iterations:
        episode_count += 1
        points = episode(DQNAgent)
        mean_history.append(points)
        if episode_count % 100 == 0:
            points_history.append(np.mean(mean_history))
        print(f'points for episode {episode_count}: {points}')
        print(f'time elapsed: {round(time.time()-start_time,3)} seconds, avg: {round(len(DQNAgent.memory)/(time.time()-start_time),0)} iterations per second \n')

    return points_history


if __name__ == "__main__":
    start_time = time.time() 
    # num_episodes = 1000
    num_learning_iterations = 1000
    points_history = run_training(num_learning_iterations)
    episodes_v = [i for i in range(int(len(points_history)))]
    env.close()

    total_t = round(time.time()-start_time,3)
    print(f'TOTAL TIME TAKEN: {total_t} seconds')
    plt.plot(episodes_v, points_history, '.')
    plt.savefig('Breakout_score_vr_epochs_#100.pdf')
    plt.show()
