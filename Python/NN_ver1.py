# Neural Network architecture.
# Input: preprocessed image of size 84 x 84 x 4.
# Layer 1 convolves 16 8x8 filters
import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model, Sequential
import gym
import numpy as np
import random as rnd

#input_shape = (80, 80, 4)

class DQN_net:
    def __init__(self, input_size, action_size):
        #Parameters
        self.actions = action_size #Size of actions space, will be the size of network output
        self.input_size = input_size

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


#TEST CODE
if __name__ == "__main__":
    env = gym.make('MsPacman-v0')
    frame = env.reset()
    state_size = env.observation_space.shape
    #state_size = (84,84,4)
    print(state_size)
    action_size = env.action_space.n #Gives a size of 9?!? change to 4!!
    print(action_size)
    test_net = DQN_net(state_size, action_size)
    '''
    # Formatting input shape for first conv2D layer
    frame, reward, is_done, _ = env.step(env.action_space.sample())
    #y = np.reshape(x, (10, 15, 1))
    #print(frame[1][1][1])
    obs = np.expand_dims(frame, axis=0)     # (Formatting issues) Making the observation the first element of a batch of inputs
    #input_tensor = np.stack((frame, frame), axis=1)
    target_f = test_net.model.predict(obs)
    #(1,210,120,3)
    '''
