from collections import deque

class DQNAgent:
    def __init__(self, input_shape, action_size):
        #self.state_size = state_size
        #self.action_size = action_size # size of 2 in CartPole caase
        self.memory = deque(maxlen=2000) #Replay Memory
        self.gamma = 0.95    # discount rate
        self.epsilon = 0.1 # exploration rate
        #self.epsilon_min = 0.01
        #self.epsilon_decay = 0.995
        #self.learning_rate = 0.001
        self.network = DQN_net(action_size, input_shape)


    def add_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        #Still have to code for when memory fills up

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            #Selects one of the possible actions randomly
            return env.action_space.sample()

        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action
