# Implements the Neural Network architecture.

from keras.models import Sequential, Input, Model
from keras.layers import Conv2D, Flatten, Dense, Convolution2D, Lambda, Multiply, convolutional, multiply
from keras.optimizers import RMSprop
from keras.backend import set_image_data_format
import gym
import numpy as np
from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # reduces verbosity of tensorflow ?


class DQN_net:
    def __init__(self, input_size, action_size,
                 batch_size=32,
                 discount_factor=0.95,
                 learning_rate=0.00025):

        # Hyper Parameters
        self.actions = action_size
        self.input_size = input_size
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Model components:
        frames_input = Input(input_size, name='frames')
        actions_input = Input((action_size,), name='mask')
        norm_frames = Lambda(lambda x: x / 255.0)(frames_input) # may need chaning qq
        conv_1 = Conv2D(16, (8, 8), strides=4, activation='relu', input_shape=input_size)(norm_frames)
        conv_2 = Conv2D(32, (4, 4), strides=2, activation='relu')(conv_1)
        flatten = Flatten()(conv_2)
        dense_1 = Dense(256, activation='relu')(flatten)
        output = Dense(action_size)(dense_1)
        masked_output = Multiply()([output, actions_input])

        # compile model:
        self.model = Model(inputs=[frames_input, actions_input], outputs=masked_output)
        optimizer = RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)  # from 'Human-level control through deep reinforcement learning'
        self.model.compile(optimizer, loss='mean_squared_error')

    def train(self, batch_states, batch_actions, batch_rewards, batch_new_states, batch_is_dones, target_network):
        '''
        EXPERIENCE_BATCH:
        an element of experieince batch looks like:
        (state, action, reward, next_state, done)
        and is a list of size batch_size

        TARGET_NETWORK:
        target_network is a DQN_net instance to generate target according
        to the DQN algorithm.
        '''
        assert type(target_network) == DQN_net
        #state_train = np.zeros((self.batch_size,) + self.input_size)
        batch_indices = np.array([i for i in range(32)])
        target_batch = np.zeros((self.batch_size,) + (self.actions,))
        action_mask_batch = np.zeros((self.batch_size,) + (self.actions,))  # hotwired actions
        open_mask = np.ones(self.actions)
        open_mask_batch = np.stack((open_mask,)*self.batch_size, axis = 0)

        output_target_pred = target_network.model.predict([batch_new_states, open_mask_batch])
        #max_q_index_pred = np.argmax(output_target_pred)
        max_q_value_pred = np.max(output_target_pred, axis=-1)
        action_mask_batch[batch_indices, batch_actions] = 1

        True_indicies = np.where(batch_is_dones == True)
        False_indicies = np.where(batch_is_dones == False)

        #For terminal transitions:
        target_batch[True_indicies,batch_actions[True_indicies]] = batch_rewards[True_indicies]
        #For non terminal states:
        target_batch[False_indicies,batch_actions[False_indicies]] = batch_rewards[False_indicies] + \
                                self.discount_factor * max_q_value_pred[False_indicies]

        #New input: batch_states
        self.model.fit([batch_states, action_mask_batch], target_batch,
                       batch_size=self.batch_size, epochs=1, verbose=0)


# TEST CODE
if __name__ == "__main__":
    from preprocess import preprocess

    env = gym.make('BreakoutDeterministic-v4')
    frame = env.reset()
    new_frame, reward, is_done, _ = env.step(env.action_space.sample())
    new_frame = preprocess(new_frame)
    state = np.stack((new_frame,)*4, axis=-1)
    state_size = state.shape



    action_size = env.action_space.n  # Gives a size of 9?!? change to 4!!
    print("state size:", state_size)
    print("action size:", action_size)
    obs = np.expand_dims(state, axis=0)     # (Formatting issues) Making the observation the first element of a batch of inputs
    print("obs shape:", obs.shape)
    test_net = DQN_net(state_size, action_size)

    # predictions
    mask_ones = np.ones(action_size)
    mask_ones = np.expand_dims(mask_ones, axis=0)
    # mask_ones[0][2] = 1
    test_target_predicted = test_net.model.predict([obs, mask_ones])
    print("Q predictions:", test_target_predicted)
    print(f"preferrable action-index: {np.argmax(test_target_predicted)} ({max(test_target_predicted[0])})")
