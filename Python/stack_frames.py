
from collections import deque
import numpy as np


class stack_frames:
    '''
    Functions takes a preprocessed frame as input.
    Basically a Deque with more functionality.
    '''
    def __init__(self, stack_size=4):

        self.stack_size = stack_size
        self.frame_stack = deque(maxlen=stack_size)

    def create_stack(self, initial_frame):
        '''
        Creates a full stack of the initial frame only.
        Input: Frame of shape: (84,84)
        Output: np array, shape: (84,84,4)
        Still needs to be exdended dimensionally to be fed to NN: (x,84,84,4)
        '''
        for i in range(self.stack_size):
            self.frame_stack.append(initial_frame)
        initial_state = np.stack((initial_frame,)*self.stack_size, axis=-1)
        return initial_state

    def get_new_state(self,new_frame):
        '''
        Adds a new frame to stack in the FIFO principle.
        Input: Frame of shape: (84,84)
        Output: np array, shape: (84,84,4)
        Still needs to be exdended dimensionally to be fed to NN: (x,84,84,4)
        '''
        self.frame_stack.append(new_frame)
        new_state = np.stack((elem for elem in self.frame_stack),axis=-1)
        return new_state
