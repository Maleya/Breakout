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
