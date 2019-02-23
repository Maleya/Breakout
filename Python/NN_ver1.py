# Neural Network architecture.

from keras.models import Sequential, Input, Model
from keras.layers import Conv2D, Flatten, Dense, Convolution2D, Lambda, Multiply, convolutional, multiply
from keras.optimizers import RMSprop
from keras.backend import set_image_data_format
# import keras
import gym
import numpy as np
from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # reduces verbosity of tensorflow ?
# set_image_data_format('channels_first')  # very experimental!!


class DQN_net:
    def __init__(self, input_size, action_size, action_mask=None,
                 batch_size=32,
                 discount_factor =0.95,
                 learning_rate = 0.00025):
        # Hyper Parameters
        self.actions = action_size
        self.input_size = input_size
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.action_mask = action_mask
        if action_mask is None:  # for legacy reasons
            self.action_mask = np.ones(action_size)


        # Model:
        frames_input = Input(input_size, name='frames')
        actions_input = Input((action_size,), name='mask')
        norm_frames = Lambda(lambda x: x / 255.0)(frames_input)
        # print("helpme",norm_frames.shape)
        conv_1 = Conv2D(16, (8, 8), strides=4, activation='relu', input_shape=input_size)(norm_frames)
        conv_2 = Conv2D(32, (4, 4), strides=2, activation='relu')(conv_1)

        flatten = Flatten()(conv_2)
        dense_1 = Dense(256, activation='relu')(flatten)
        output = Dense(action_size)(dense_1)
        masked_output = Multiply()([output, actions_input])

        self.model = Model(inputs=[frames_input, actions_input], outputs=masked_output)
        optimizer = RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)  # from 'Human-level control through deep reinforcement learning'
        self.model.compile(optimizer, loss='mean_squared_error')

    # def train(self, experience_batch, target_network):
    #     '''
    #     EXPERIENCE_BATCH:
    #     an element of experieince batch looks like:
    #     (state, action, reward, next_state, done)
    #     and is a list of size batch_size

    #     TARGET_NETWORK:
    #     target_network is a DQN_net instance to generate target according
    #     to the DQN algorithm.
    #     '''
    #     assert type(target_network) == DQN_net
    #     state_train = np.zeros((self.batch_size,) + self.input_size)
    #     target_train = np.zeros((self.batch_size,) + (self.actions,))
    #     for i, experience in enumerate(experience_batch):

    #         state_train[i] = experience[0]
    #         action_train = experience[1]
    #         reward_train = experience[2]
    #         next_state_train = experience[3]
    #         is_done = experience[4]

    #         output_target_pred = target_network.model.predict(next_state_train)
    #         output_current_state = self.model.predict(state_train)

    #         # output_target_pred_shape = [[q_action_1, ... ,q_action_n]]
    #         for k, elem in enumerate(output_current_state[0]):
    #             target_train[i][k] = elem

    #         max_q_value_pred = np.max(output_target_pred[0])
    #         # max_q_action = np.argmax(output_target_pred[0])

    #         if is_done is True:
    #             target_train[i][action_train] = reward_train
    #         else:
    #             target_train[i][action_train] = reward_train + \
    #                                     self.discount_factor * max_q_value_pred  # output_target_pred[0][max_q_action]

        # self.model.fit(state_train,
    #                    target_train,
    #                    batch_size=self.batch_size,
    #                    epochs=1,
    #                    verbose=0)


# TEST CODE
if __name__ == "__main__":
    from preprocess_BO import pre_process_BO

    env = gym.make('Breakout-v0')
    frame = env.reset()
    new_frame, reward, is_done, _ = env.step(env.action_space.sample())
    new_frame = pre_process_BO(new_frame)
    state = np.stack((new_frame,)*4, axis=-1)
    state_size = state.shape



    action_size = env.action_space.n  # Gives a size of 9?!? change to 4!!
    print("state size:", state_size)
    print("action size:", action_size)
    obs = np.expand_dims(state, axis=0)     # (Formatting issues) Making the observation the first element of a batch of inputs
    print("obs shape:", obs.shape)
    test_net = DQN_net(state, action_size)

    # print('action mask:',test_net.action_mask)

    # test = [1 for i in range(action_size)]
    # predictions
    # test_target_predicted = test_net.model.predict(state)
    # print("Q predictions:", test_target_predicted)
    # print(f"preferrable action-index: {np.argmax(test_target_predicted)} ({max(test_target_predicted[0])})")
    # ------------------------
    # new_frame, reward, is_done, _ = env.step(env.action_space.sample())

    # # input_tensor = np.stack((frame, frame), axis=1)
    # target_f = test_net.model.predict([obs, np.ones(action_size)])
    # # print(target_f)



# print("==========this is for testing purposes========")

#     env = gym.make('MsPacman-v0')
#     frame = env.reset()
#     frame, reward, is_done, _ = env.step(env.action_space.sample())

#     #dimensions
#     state_size = env.observation_space.shape
#     action_size = env.action_space.n #Gives a size of 9?!? change to 4!!
#     print("state size:",state_size)
#     print("action size:",action_size,)
#     test_net = Neuralnet(state_size, action_size)

#     # Predictions
#     obs = np.expand_dims(frame, axis=0)
#     test_target_predicted = test_net.model.predict(obs)
#     print("Q predictions:",test_target_predicted)
#     print(f"preferrable action-index: {np.argmax(test_target_predicted)} ({max(test_target_predicted[0])})")
#     # print(max(test_target_predicted[0]))

#     print("==================================================")
