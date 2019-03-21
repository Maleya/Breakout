"""
This runs our main algorithm, and is the highest in the code heirachy import-wise.

note:
num_learning_iterations starts counting after we filled agent memory with agent.batch_size*[num]
Before running a fresh run:
    clean out latest_epsilon and plot_data.csv
"""
import gym
import numpy as np
import time
import csv
import matplotlib
matplotlib.use("TkAgg")  # for mac users
from matplotlib import pyplot as plt
from os import environ, path
# from keras.models import load_model
# from NeuralNet import DQN_net
from Agent import DQN_Agent
from stack_frames import stack_frames
from preprocess import preprocess
environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # reduces verbosity of tensorflow?
env = gym.make('BreakoutDeterministic-v4')


# SETTINGS & PARAMETERS --------------------------------------------------
saved_NN_weights = "saved_weights_new_run_test1.h5"  # varaiable names set here
saved_NN_target_weights = "target_saved_weights_new_run_test1.h5"
saved_epsilon = "latest_epsilon_new_run_test1.csv"
saved_scores = "plot_data.csv"

num_learning_iterations = 1000000
learning_delay = 50000

#DATA GATHERING
prel_history = []
points_history = []


def episode(agent):
    '''An episode constitues one normal run of a game.
    Takes an Agent as input and returns the episode points'''

    assert type(agent) == DQN_Agent
    points = 0
    learning_steps = 0
    ep_start_time = time.time()

    # PREP A FRAME
    initial_frame = env.reset()
    initial_frame = preprocess(initial_frame)
    stacked_frames = stack_frames()
    state = stacked_frames.create_stack(initial_frame)
    is_done = False

    # RUN A FULL EPISODE:
    while not is_done:
        if agent.video is True:
            env.render()

        action = agent.get_action(state)
        new_frame, reward, is_done, _ = env.step(action)
        points += reward
        # Procecessing images to 4D tensor for the conv_2D input
        new_frame = preprocess(new_frame)  # Breakout
        new_state = stacked_frames.get_new_state(new_frame)
        #implement new memory
        if agent.iteration_count % 4 == 0:
            agent.memory.add(state, action, reward, new_state, is_done)
            # Network weights update: starts after delay.
            mem_len = agent.memory.memory_len
            if mem_len >= learning_delay:
                batch_states, batch_actions, batch_rewards, batch_new_states, batch_is_dones = agent.memory.sample_batch()
                agent.network.train(batch_states, batch_actions, batch_rewards, batch_new_states, batch_is_dones, agent.target_network)
                agent.learning_count += 1
                learning_steps += 1
                if agent.learning_count % agent.learning_count_max == 0:
                    agent.reset_target_network()

                # decaying epsilon.
                if agent.epsilon > agent.min_epsilon:
                    agent.epsilon *= agent.epsilon_decay
            #SAVE POINTS IN HISTORY LIST:

            if agent.learning_count % 50000 == 0 and agent.learning_count != 0:
                points_history.append(np.mean(prel_history))
                print(prel_history)
                row = [agent.learning_count, np.mean(prel_history)]
                with open(f'./data/{saved_scores}', 'a', newline='') as csvFile:
                    writer = csv.writer(csvFile)
                    writer.writerow(row)
                csvFile.close()
                prel_history.clear()
        state = new_state
        agent.iteration_count += 1

    prel_history.append(points)
    # ----PRINTS FOR TESTING ----------------------
    print(f'Did {learning_steps} minibatch updates in {round(time.time()-ep_start_time,3)} seconds with ε = {round(agent.epsilon,3)}')

    # ---------------------------------------------
    return points

def run_training(num_learning_iterations):
    '''This trains our model along with some house-keeping features'''
    frame = env.reset()
    frame = preprocess(frame)
    state = np.stack((frame,)*4, axis=-1)

    episode_count = 0
    state_size = state.shape
    action_size = env.action_space.n

    DQNAgent = DQN_Agent(state_size, action_size,
                         batch_size=32,
                         discount_factor=0.99,
                         learning_rate=0.00025,
                         epsilon=1,
                         epsilon_decrease_rate=0.999997697417558,  # becomes 0.1 after 10**6 learning iterations
                         min_epsilon=0.1,
                         video=False,
                         epsilon_linear=True)

    # LOAD FILES ---------------------------------------------------------------------
    if path.isfile(f'./data/{saved_NN_weights}'):
        DQNAgent.network.model.load_weights(f'./data/{saved_NN_weights}')
        print(f"{saved_NN_weights} loaded successfully!")

    else:
        print(f"IMPORT WARNING: '{saved_NN_weights}' was not found!")
        print('starting fresh...')

    if path.isfile(f'./data/{saved_NN_target_weights}'):
        DQNAgent.target_network.model.load_weights(f'./data/{saved_NN_target_weights}')
        print(f"{saved_NN_target_weights} loaded successfully!")

    else:
        print(f"IMPORT WARNING: '{saved_NN_target_weights}' was not found!")
        print('starting fresh...')

    if path.isfile(f'./data/{saved_epsilon}'):
        with open(f'./data/{saved_epsilon}', 'rb') as eps:
            eps = eps.read().decode().strip()
            DQNAgent.epsilon = float(eps)
        print(f"{saved_epsilon} loaded successfully!")

    else:
        print(f'IMPORT WARNING: {saved_epsilon} was not found!')
    print("\n\n")

    # named section  ---------------------------------------------------------------------
    while DQNAgent.learning_count < num_learning_iterations:
        points = episode(DQNAgent)
        episode_count += 1
        # PROGRESS PRINTS
        print(f'points for episode {episode_count}: {points}')
        #print(f'time elapsed: {round(time.time()-start_time,3)} seconds, avg: {round(DQNAgent.memory)/(time.time()-start_time),0)} iterations per second ')
        print(f"learning iterations: {round(DQNAgent.learning_count/num_learning_iterations*100,3)}% done. [{DQNAgent.learning_count}/{num_learning_iterations}] \n")
    #csvFile.close()
    return points_history, DQNAgent


if __name__ == "__main__":
    start_time = time.time()
    points_history, DQNAgent = run_training(num_learning_iterations)
    env.close()

    # SAVE FILES
    DQNAgent.network.model.save_weights(f'./data/{saved_NN_weights}')
    DQNAgent.target_network.model.save_weights(f'./data/{saved_NN_target_weights}')
    with open(f'./data/{saved_epsilon}', 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow([DQNAgent.epsilon])
    csvFile.close()

    # plots and time-keeping
    episodes_v = [i for i in range(int(len(points_history)))]
    total_t = round(time.time()-start_time, 3)
    print(f'TOTAL TIME TAKEN: {round(total_t/3600,3)} hours')
    print(f'TOTAL TIME TAKEN: {total_t} seconds')

    plt.plot(episodes_v, points_history, '.')
    plt.xlabel('Number of Played Game Epochs.')
    plt.ylabel('Average Game Score.')
    plt.savefig('./data/Breakout_score_vr_epochs_#100.pdf')
    plt.show()
