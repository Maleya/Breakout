# Neural Network architecture.

from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense
import gym
import numpy as np
import random



class DQN_net:
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

        # Sequential() creates the foundation of the layers.
        self.model = Sequential()
        # First convolutional layer
        # The first hidden layer convolves 16 8×8 filters
        # with stride 4 with the input image and applies a rectifier nonlinearity."
        self.model.add(Conv2D(16, (8, 8), strides=4,
                              activation='softplus',
                              input_shape= input_size))

        # Second convolutional layer
        # The second hidden layer convolves 32 4×4 filters
        # with stride 2, again followed by a rectifier nonlinearity."
        self.model.add(Conv2D(32, (4,4), strides=2,
                              activation='softplus'))

        # Flatten the convolution output
        self.model.add(Flatten())

        # First dense layer
        self.model.add(Dense(256, activation='softplus'))

        # Output layer
        self.model.add(Dense(self.actions))

        # Create the model based on the information above
        self.model.compile(loss='mean_squared_error',
                           optimizer='rmsprop',
                           metrics=['accuracy'])

    def train(self, experience_batch, target_network):
        ''' Experience_batch is a list of size "batch_size" with elements
        randomly drawn from the Replay-memory.
        elements = (state, action, reward, next_state, done)
        -target_network is a DQN_net instance to generate target according
        to the DQN algorithm.
        '''

        state_train = np.zeros((self.batch_size,) + self.input_size)
        target_train = np.zeros((self.batch_size,) + (self.actions,))
        #print('target_train=',target_train)
        for i_train, experience in enumerate(experience_batch):
            # Inputs are the states
            #print('state_train is' + str(experience[0]))
            state_train[i_train] = experience[0]

            action_train = experience[1]
            reward_train = experience[2]
            next_state_train = experience[3]
            is_done = experience[4]

            output_target_pred = target_network.model.predict(next_state_train)
            #print('output_target pred=',output_target_pred)
            #next_q_value_pred = np.max(output_target_pred)
            max_q_action = np.argmax(output_target_pred)
            #print(max_q_action)
            #print(output_target_pred)

            #BEllMAN..?
            if is_done == True:
                target_train[i_train][max_q_action] = reward_train
            else:
                target_train[i_train][max_q_action] = reward_train + \
                                        self.discount_factor * output_target_pred[0][max_q_action]
        #Need to do .ravel() / .squeeze()?? on state + target
        #state_train = state_train.ravel()
        #state_train = np.asarray(state_train).squeeze()
        #target_train = np.asarray(target_train).squeeze()
        # Train the model for one epoch
        #print('New target_train' + str(target_train))
        self.model.fit(state_train,
                       target_train,
                       batch_size = self.batch_size,
                       epochs=10)


#TEST CODE
if __name__ == "__main__":
    env = gym.make('MsPacman-v0')
    frame = env.reset()
    state_size = env.observation_space.shape
    #state_size = (84,84,4)
    print("state size:",state_size)
    action_size = env.action_space.n #Gives a size of 9?!? change to 4!!
    print("action size:",action_size)
    test_net = DQN_net(state_size, action_size)

    # Formatting input shape for first conv2D layer
    frame, reward, is_done, _ = env.step(env.action_space.sample())
    #y = np.reshape(x, (10, 15, 1))
    #print(frame[1][1][1])
    obs = np.expand_dims(frame, axis=0)     # (Formatting issues) Making the observation the first element of a batch of inputs
    #input_tensor = np.stack((frame, frame), axis=1)
    #print(obs)
    target_f = test_net.model.predict(obs)
    #(1,210,120,3)
