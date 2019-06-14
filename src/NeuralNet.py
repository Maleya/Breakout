# Implements the Neural Network architecture.

from keras.models import Sequential, Input, Model
from keras.layers import Conv2D, Flatten, Dense, Convolution2D, Lambda, Multiply, convolutional, multiply
from keras.optimizers import RMSprop
from keras.backend import set_image_data_format
from keras.initializers import VarianceScaling
import gym
import numpy as np
from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # reduces verbosity of tensorflow ?


class DQN_net:
    """
    info
    """
    def __init__(self, input_size, action_size,
                 batch_size=32,
                 discount_factor=0.95,
                 learning_rate=0.00025,
                 gradient_momentum=0.95):

        # Hyper Parameters
        self.actions = action_size
        self.input_size = input_size
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Model components:
        initializer = VarianceScaling(scale=2.0, mode='fan_in', distribution='normal', seed=None)
        frames_input = Input(input_size, name='frames')
        actions_input = Input((action_size,), name='mask')
        # norm_frames = Lambda(lambda x: x / 255.0)(frames_input) # now in preprocessing
        conv_1 = Conv2D(16, (8, 8), strides=4, activation='relu', input_shape=input_size,kernel_initializer=initializer)(frames_input)
        conv_2 = Conv2D(32, (4, 4), strides=2, activation='relu', kernel_initializer=initializer)(conv_1)
        #conv_3 = Conv2D(64, (3, 3), strides=1, activation='relu', kernel_initializer=initializer)(conv_2)
        flatten = Flatten()(conv_2)
        dense_1 = Dense(256, activation='relu')(flatten)
        output = Dense(action_size)(dense_1)
        masked_output = Multiply()([output, actions_input])

        # compile model:
        self.model = Model(inputs=[frames_input, actions_input], outputs=masked_output)
        optimizer = RMSprop(lr=learning_rate, rho=gradient_momentum, epsilon=0.01, clipnorm=1.)  # from 'Human-level control through deep reinforcement learning'
        self.model.compile(optimizer, loss='mean_squared_error')

    def train(self, batch_states, batch_actions, batch_rewards, batch_new_states, batch_is_dones, target_network):
        '''
        Preforms a minibatch update of neural network.

        INPUT SHAPES:
        batch_states: (32,84,84,4)
        batch_actions:(32,action_size)
        batch_rewards:(32)
        batch_new_states:(32,84,84,4)
        batch_is_dones:(32)
        target_network: DQN_net instance to generate target according
        to the DQN algorithm.
        '''
        assert type(target_network) == DQN_net
        # state_train = np.zeros((self.batch_size,) + self.input_size)
        batch_indices = np.array([i for i in range(self.batch_size)])
        target_batch = np.zeros((self.batch_size,) + (self.actions,))
        action_mask_batch = np.zeros((self.batch_size,) + (self.actions,))  # hotwired actions
        open_mask = np.ones(self.actions)
        open_mask_batch = np.stack((open_mask,)*self.batch_size, axis = 0)

        output_target_pred = target_network.model.predict([batch_new_states, open_mask_batch])
        # max_q_index_pred = np.argmax(output_target_pred)
        max_q_value_pred = np.max(output_target_pred, axis=-1)
        action_mask_batch[batch_indices, batch_actions] = 1

        True_indicies = np.where(batch_is_dones == True)
        False_indicies = np.where(batch_is_dones == False)

        # For terminal transitions:
        target_batch[True_indicies, batch_actions[True_indicies]] = batch_rewards[True_indicies]
        # For non-terminal transitions:
        target_batch[False_indicies, batch_actions[False_indicies]] = batch_rewards[False_indicies] + \
                                self.discount_factor * max_q_value_pred[False_indicies]

        # New input: batch_states
        self.model.fit([batch_states, action_mask_batch], target_batch,
                       batch_size=self.batch_size, epochs=1, verbose=0)
