import gym
import numpy as np
import random as rnd
from collections import deque
from cart_pole_network import cart_pole_network as cpn
env = gym.make('CartPole-v1')

class cart_pole_agent:
    def __init__(self, state_size, action_size,
                 batch_size = 1,
                 discount_factor = 0.95,
                 learning_rate = 0.00025,
                 epsilon=1,
                 epsilon_decrease_rate=0.99,
                 min_epsilon=0.1,
                 video = False,
                 epsilon_linear = True):
        #PARAMETERS
        self.video = video

        self.state_size = state_size
        self.action_size = action_size # should be 4 for pacman
        self.epsilon = epsilon # Exploration rate
        self.epsilon_decay = epsilon_decrease_rate
        self.min_epsilon = min_epsilon

        self.discount_factor = discount_factor  # Discount factor of the MDP (gamma)
        self.learning_rate = learning_rate  # Learning rate (alpha)
        self.batch_size = batch_size
        #For reseting target network
        self.learning_count = 0
        self.learning_count_max = 1000
        #Replay Memory for bootstrapping
        self.memory = deque(maxlen=100000)

        #Neural Networks for the DQN:
        #Main Networks that continuisly choose actions.
        self.network = cpn(self.state_size, self.action_size,
                     batch_size = self.batch_size,
                     discount_factor = self.discount_factor,
                     learning_rate = self.learning_rate)

        #Network to predict target in training algorithm
        self.target_network = cpn(self.state_size, self.action_size,
                     batch_size = self.batch_size,
                     discount_factor = self.discount_factor,
                     learning_rate = self.learning_rate)

    def add_experience(self, state, action, reward, next_state, done):
        '''If len(memory) = maxlen it will pop the oldest data from left
        and add the new data at the end of the list.
        '''
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            #Selects one of the possible actions randomly
            return env.action_space.sample()

        act_values = self.network.model.predict(state)
        #Shape: [ [q_action_1, ..., q_action_n] ]
        # returns index corresponding to chosen action
        return np.argmax(act_values[0])

    def sample_experience(self):
        return rnd.sample(self.memory, self.batch_size)

    def reset_target_network(self):
        """
        Updates the target DQN with the current weights of the main DQN.
        """
        self.target_network.model.set_weights(self.network.model.get_weights())


#TEST CODE
if __name__ == "__main__":
    env = gym.make('MsPacman-v0')
    frame = env.reset()
    state_size = env.observation_space.shape
    action_size = env.action_space.n #Gives a size of 9?!? change to 4!
    test_agent = DQN_Agent(state_size, action_size)
    state = np.expand_dims(frame, axis=0)
    #-----get_action-----
    action = test_agent.get_action(state)

    #---- add_experience ----
    new_frame, reward, is_done, _ = env.step(action)
    new_state = np.expand_dims(new_frame, axis=0)
    test_agent.add_experience(state,action,reward,new_state,is_done)
    experience_batch = test_agent.sample_experience()
    #print(experience_batch)
    test_agent.network.train(experience_batch, test_agent.target_network)