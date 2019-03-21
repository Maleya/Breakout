import gym
import numpy as np


class Replay_Memory:
    """
    Replay memory that saves the last 250 000 transitions.
    One stores state is a stack of four consecutive preprocessed frames.
    """
    def __init__(self, maxlen=250000, batch_size=32, frame_height=84,
                frame_width=84, num_stacked_frames=4):

        self.maxlen = maxlen
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.num_stacked_frames = num_stacked_frames
        self.batch_size = batch_size
        self.memory_len = 0
        self.count = 0

        # Pre-allocate memory
        self.states = np.empty((self.maxlen, self.frame_height, self.frame_width,
                                self.num_stacked_frames), dtype=np.uint8)
        self.actions = np.empty(self.maxlen, dtype=np.uint8)
        self.rewards = np.empty(self.maxlen, dtype=np.uint8)
        self.new_states = np.empty((self.maxlen, self.frame_height, self.frame_width,
                                self.num_stacked_frames), dtype=np.uint8)
        self.terminal_flags = np.empty(self.maxlen, dtype=np.bool)

        self.indices = np.empty(self.batch_size, dtype=np.int32)

    def add(self,state, action, reward, new_state, is_done):
        '''
        Input shapes:
        state: (4,84,84)
        action: scalar
        reward: scalar
        new state: (4,84,84)
        is_done: booleon

        '''
        assert state.shape == (self.frame_height, self.frame_width, self.num_stacked_frames)
        index = self.count % self.maxlen
        assert self.states[index].shape == state.shape

        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.new_states[index] = new_state
        self.terminal_flags[index] = is_done

        self.count += 1
        self.memory_len = min(self.count, self.maxlen)

    def sample_batch(self):
        '''
        Returns 5 lists of size 32 (batch size), which are corresponding to:
        state, action, reward, new_state, is_done

        '''
        assert self.memory_len  >= self.batch_size
        batch_indices = np.random.randint(self.memory_len, size=self.batch_size)
        self.indices = batch_indices
        return self.states[batch_indices], self.actions[batch_indices], self.rewards[batch_indices], self.new_states[batch_indices], self.terminal_flags[batch_indices]


# TEST CODE
if __name__ == "__main__":
    from preprocess import preprocess
    from Agent import DQN_Agent

    Replay_Memory = Replay_Memory()
    env = gym.make('BreakoutDeterministic-v4')
    frame = env.reset()
    for i in range(1000):
        action = env.action_space.sample()
        new_frame_raw, reward, is_done, _ = env.step(action)
        if is_done == True:
            frame = env.reset()
        # print(is_done)
        # env.render()
        new_frame = preprocess(new_frame_raw)
        state = np.stack((new_frame,)*4, axis=-1)
        Replay_Memory.add(state, action, reward, state, is_done)

    batch_states, batch_actions, batch_rewards, batch_new_states, batch_is_dones = Replay_Memory.sample_batch()
    # print(f'shape of batch_states is: {batch_states.shape}')
    # print(f'First element in batch_states is: {batch_states[21][21][20]}')
    # print(f'Last state: {new_frame_raw}')
    # print(f'Last state processed: {new_frame}')
    state_size = state.shape
    # print(f'State shape is: {state_size}')

    state_size = state.shape
    action_size = env.action_space.n

    Agent = DQN_Agent(state_size, action_size)
    # open_mask = np.array([0,1,0,0])
    open_mask = np.ones(action_size)
    open_mask = np.stack((open_mask,)*32, axis=0)
    output = Agent.network.model.predict([batch_new_states, open_mask])
    print(output)
    max_q_pred = np.max(output, axis=-1)
    print(f'max output values is: {max_q_pred}')
    # print(len(max_q_index_pred))
    action_mask_array = np.zeros((32,) + (4,))
    # print(action_mask_array)
    action_mask_array[np.array([i for i in range(32)]),batch_actions] = 1
    target_batch = np.zeros((32,) + (4,))
    True_indicies = np.where(batch_is_dones == True)
    target_batch[True_indicies, batch_actions[True_indicies]] = batch_rewards[True_indicies]
    # print(batch_rewards)
    # print(True_indicies)
    # print(target_batch)
    False_indicies = np.where(batch_is_dones == False)
    target_batch[False_indicies, batch_actions[False_indicies]] = batch_rewards[False_indicies] + \
                            0.99 * max_q_pred[False_indicies]
    print(max_q_pred[False_indicies])
    print(target_batch)
    print(Replay_Memory.indices)
    print(f'the false indicies: {False_indicies}')
    print(f'True indicies: {True_indicies}')
