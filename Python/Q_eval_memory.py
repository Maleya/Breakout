import numpy as np

class Q_eval_memory:
    """
    Memory for evaluation of Q_value predictions through out the learning process.
    """
    def __init__(self, maxlen=2000, frame_height=84,
                frame_width=84, num_stacked_frames=4):

        self.maxlen = maxlen
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.num_stacked_frames = num_stacked_frames
        self.memory_len = 0
        self.count = 0

        # Pre-allocate memory
        self.states = np.empty((self.maxlen, self.frame_height, self.frame_width,
                                self.num_stacked_frames), dtype=np.uint8)
    def add(self,state):
        '''
        Input shapes: state: (4,84,84)
        '''
        assert state.shape == (self.frame_height, self.frame_width, self.num_stacked_frames)
        index = self.count % self.maxlen
        assert self.states[index].shape == state.shape

        self.states[index] = state
        self.count += 1
        self.memory_len = min(self.count, self.maxlen)


if __name__ == "__main__":
    from preprocess import preprocess
    from Agent import DQN_Agent
    import gym

    Q_eval_memory = Q_eval_memory()
    env = gym.make('BreakoutDeterministic-v4')
    frame = env.reset()
    for i in range(500):
        action = env.action_space.sample()
        new_frame_raw, reward, is_done, _ = env.step(action)
        if is_done == True:
            frame = env.reset()
        new_frame = preprocess(new_frame_raw)
        state = np.stack((new_frame,)*4, axis=-1)
        Q_eval_memory.add(state)

    print(Q_eval_memory.states.shape)
