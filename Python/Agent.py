"""
We construct the learning agent that will house two Neuralnet instances. 

"""


import gym
import numpy as np
import random as rnd
from collections import deque
from Neuralnet import Neuralnet
env = gym.make('MsPacman-v0')


class Agent:
    def __init__(self, state_size, action_size,
                 batch_size=1,
                 discount_factor=0.95,
                 learning_rate=0.00025,
                 epsilon=0.1):
        #PARAMETERS
        self.state_size = state_size
        self.action_size = action_size  # should be 4 for pacman
        self.epsilon = epsilon  # Exploration rate
        self.discount_factor = discount_factor  # Discount factor of the MDP (gamma)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.learning_count = 0
        self.learning_count_max = 1000
 
        self.memory = deque(maxlen=2000)  # Replay Memory for bootstrapping

        # Main NN that continuously chooses actions.
        self.network = Neuralnet(self.state_size, self.action_size,
                     batch_size = self.batch_size,
                     discount_factor = self.discount_factor,
                     learning_rate = self.learning_rate,
                     epsilon = self.epsilon)

        # Network to predict target in training algorithm
        self.target_network = Neuralnet(self.state_size, self.action_size,
                     batch_size = self.batch_size,
                     discount_factor = self.discount_factor,
                     learning_rate = self.learning_rate,
                     epsilon = self.epsilon)

    def get_action(self, state):
        """
       Returns a random action from the actionspace with probability ε.
       else: return return action with highest q value.
        """

        if np.random.rand() <= self.epsilon:
            return env.action_space.sample()  # random action chosen

        else: 
            act_values = self.network.model.predict(state)  # Shape: [ [q_action_1, ..., q_action_n] ]
            return np.argmax(act_values[0])

    def add_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample_experience(self):
        return rnd.sample(self.memory, self.batch_size)

    def reset_target_network(self):
        """
        Updates the target_network with weights of the main NN.
        """
        self.target_network.model.set_weights(self.network.model.get_weights())


# TEST CODE
if __name__ == "__main__":
    print("==========this is for testing purposes========")

    env = gym.make('MsPacman-v0')
    frame = env.reset()
    state_size = env.observation_space.shape
    action_size = env.action_space.n #Gives a size of 9?!? change to 4!
    test_agent = Agent(state_size, action_size)
    state = np.expand_dims(frame, axis=0)
    
    #-----get_action-----
    action = test_agent.get_action(state)

    #---- add_experience ----
    new_frame, reward, is_done, _ = env.step(action)
    new_state = np.expand_dims(new_frame, axis=0)
    test_agent.add_experience(state,action,reward,new_state,is_done)
    experience_batch = test_agent.sample_experience()
    #  print(experience_batch)
    test_agent.network.train(experience_batch, test_agent.target_network)

    print("==================================================")