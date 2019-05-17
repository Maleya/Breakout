import gym
import numpy as np
import random as rnd
from collections import deque
from NeuralNet import DQN_net
from Replay_Memory import Replay_Memory
env = gym.make('BreakoutDeterministic-v4')


class DQN_Agent:
    """
    info
    """
    def __init__(self, state_size, action_size,
                 batch_size=1,
                 discount_factor=0.99,
                 learning_rate=0.00025,
                 epsilon=1,
                 epsilon_decrease_rate=0.99,
                 min_epsilon=0.1,
                 Replay_Memory_size=250000,
                 target_upd_freq=10000,
                 gradient_momentum=0.95,
                 video=False,
                 epsilon_linear=True):

        # PARAMETERS
        self.video = video

        self.state_size = state_size
        self.action_size = action_size  # should be 4 for pacman
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decrease_rate
        self.min_epsilon = min_epsilon

        self.discount_factor = discount_factor  # Discount factor of the MDP (gamma)
        self.learning_rate = learning_rate  # Learning rate (alpha)
        self.batch_size = batch_size
        #  For reseting target network
        self.iteration_count = 0
        self.learning_count = 0
        self.learning_count_max = target_upd_freq
        # Replay Memory for bootstrapping
        #self.memory = deque(maxlen=500000)
        self.memory = Replay_Memory(maxlen=Replay_Memory_size)
        self.q_val_memory = 0
        # Neural Networks for the DQN:
        # Main Networks that continuisly choose actions.
        self.network = DQN_net(self.state_size, self.action_size,
                     batch_size=self.batch_size,
                     discount_factor=self.discount_factor,
                     learning_rate=self.learning_rate,
                     gradient_momentum=gradient_momentum)

        # Network to predict target in training algorithm
        self.target_network = DQN_net(self.state_size, self.action_size,
                     batch_size=self.batch_size,
                     discount_factor=self.discount_factor,
                     learning_rate=self.learning_rate,
                     gradient_momentum=gradient_momentum)

    def get_action(self, state):
        """
        Returns a random action with probability epsilon.
        Otherwise, returns argmax of the Q values predicted
        by the DQN_net.
        """

        if np.random.rand() <= self.epsilon:
            # Selects one of the possible actions randomly
            return env.action_space.sample()

        open_mask = np.ones(self.action_size)  # make an open mask
        open_mask = np.expand_dims(open_mask, axis=0)
        exp_state = np.expand_dims(state, axis=0)
        act_values = self.network.model.predict([exp_state, open_mask])
        # Shape: [ [q_action_1, ..., q_action_n] ]
        # returns index corresponding to chosen action
        return np.argmax(act_values[0])

    def update_target_network(self):
        """
        Updates the target DQN with the current weights of the main DQN.
        """
        self.target_network.model.set_weights(self.network.model.get_weights())

    def q_val_for_plot(self):
        assert type(self.q_val_memory) == np.ndarray 
        open_mask = np.ones(self.action_size)
        open_mask_batch = np.stack((open_mask,)*5000, axis = 0)
        q_val_for_all_states = self.network.model.predict([self.q_val_memory,open_mask_batch])
        max_q_values_for_all_states = np.max(q_val_for_all_states, axis=-1)
        return np.mean(max_q_values_for_all_states)

# TEST CODE
if __name__ == "__main__":
    env = gym.make('MsPacman-v0')
    frame = env.reset()
    state_size = env.observation_space.shape
    action_size = env.action_space.n  # Gives a size of 9?!? change to 4!
    test_agent = DQN_Agent(state_size, action_size)
    state = np.expand_dims(frame, axis=0)
    # -----get_action-----
    action = test_agent.get_action(state)

    # ---- add_experience ----
    new_frame, reward, is_done, _ = env.step(action)
    new_state = np.expand_dims(new_frame, axis=0)
    test_agent.add_experience(state, action, reward, new_state, is_done)
    experience_batch = test_agent.sample_experience()
    # print(experience_batch)
    test_agent.network.train(experience_batch, test_agent.target_network)
