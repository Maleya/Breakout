
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense
import gym
import numpy as np
import random



class cart_pole_network:
    def __init__(self, input_size, action_size,
                 batch_size = 32,
                 discount_factor = 0.95,
                 learning_rate = 0.00025):
        #Hyper Parameters
        self.actions = action_size
        self.input_size = input_size
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Sequential() creates the foundation of the layers.
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim = self.input_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.actions, activation='linear'))
        model.compile(loss='mean_squared_error',
                           optimizer='rmsprop',
                           metrics=['accuracy'])
        #model.compile(loss='mse',
        #              optimizer=Adam(lr=self.learning_rate))
        return model

    def train(self, experience_batch, target_network):
        ''' Experience_batch is a list of size "batch_size" with elements
        randomly drawn from the Replay-memory.
        elements = (state, action, reward, next_state, done)
        -target_network is a DQN_net instance to generate target according
        to the DQN algorithm.
        '''

        state_train = np.zeros((self.batch_size,) + self.input_size)
        target_train = np.zeros((self.batch_size,) + (self.actions,))
        for i_train, experience in enumerate(experience_batch):

            state_train[i_train] = experience[0]
            action_train = experience[1]
            reward_train = experience[2]
            next_state_train = experience[3]
            is_done = experience[4]

            output_target_pred = target_network.model.predict(next_state_train)
            output_current_state = self.model.predict(state_train)

            #output_target_pred_shape = [[q_action_1, ... ,q_action_n]]
            for k,elem in enumerate(output_current_state[0]):
                target_train[i_train][k] = elem

            max_q_value_pred = np.max(output_target_pred[0])
            #max_q_action = np.argmax(output_target_pred[0])

            if is_done == True:
                target_train[i_train][action_train] = reward_train
            else:
                target_train[i_train][action_train] = reward_train + \
                                        self.discount_factor * max_q_value_pred #output_target_pred[0][max_q_action]

        self.model.fit(state_train,
                       target_train,
                       batch_size = self.batch_size,
                       epochs=1,
                       verbose=0)
