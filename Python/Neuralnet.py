"""
This is where we house the functionality for the neural network 

still to be fixed: A
ction space: 9 or 4? 
Epochs set to default at 10

"""

from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense
import gym
import numpy as np
import random


class Neuralnet:

    def __init__(self, input_size, action_size,
                 batch_size = 32,
                 discount_factor = 0.95,
                 learning_rate = 0.00025,
                 epsilon = 0.1):

        #Hyper Parameters
        self.actions = action_size
        self.input_size = input_size
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.epsilon = epsilon  # Param for exploration
        self.batch_size = batch_size


        # Model
        self.model = Sequential()
        self.model.add(Conv2D(16, (8, 8), strides=4,
                              activation='softplus',
                              input_shape= input_size))

        self.model.add(Conv2D(32, (4,4), strides=2,
                              activation='softplus'))

        self.model.add(Flatten())
        self.model.add(Dense(256, activation='softplus'))
        self.model.add(Dense(self.actions))
        self.model.compile(loss='mean_squared_error',
                           optimizer='rmsprop',
                           metrics=['accuracy'])


    def train(self, experience_batch, target_network, epochs=1):
        """
        We train the exisiting model with another instance of the same class named target_network.

        """
        assert type(target_network) == Neuralnet
        state_train = np.zeros((self.batch_size,) + self.input_size)
        target_train = np.zeros((self.batch_size,) + (self.actions,))

        for i, experience in enumerate(experience_batch):
            # an element of experieince batch looks like:
            # (state, action, reward, next_state, done)
            # and is of size batch_size

            state_train[i] = experience[0]
            action_train = experience[1]
            reward_train = experience[2]
            next_state_train = experience[3]
            is_done = experience[4]

            output_target_predicted = target_network.model.predict(next_state_train)
            max_q_action = np.argmax(output_target_predicted)

            # apply reward according to is_done:
            if is_done == True: 
                target_train[i][max_q_action] = reward_train
            else:
                target_train[i_train][max_q_action] = reward_train + \
                                        self.discount_factor * output_target_predicted[0][max_q_action]
                                        # output_target_predicted has shape = (1, 210, 160, 3)

            self.model.fit(state_train, target_train, batch_size=self.batch_size, epochs=epochs,verbose=0)


if __name__ == "__main__":
    print("==========this is for testing purposes========")

    env = gym.make('MsPacman-v0')
    frame = env.reset()
    frame, reward, is_done, _ = env.step(env.action_space.sample())

    #dimensions
    state_size = env.observation_space.shape
    action_size = env.action_space.n #Gives a size of 9?!? change to 4!!
    print("state size:",state_size)
    print("action size:",action_size,)
    test_net = Neuralnet(state_size, action_size)

    # Predictions
    obs = np.expand_dims(frame, axis=0)
    test_target_predicted = test_net.model.predict(obs)
    print("Q predictions:",test_target_predicted)
    print(f"preferrable action-index: {np.argmax(test_target_predicted)} ({max(test_target_predicted[0])})")
    # print(max(test_target_predicted[0]))

    print("==================================================")

