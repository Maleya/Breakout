
from collections import deque
import numpy as np


class stack_frames:
    '''
    Functions takes a preprocessed frame as input.
    '''
    def __init__(self, stack_size = 4):

        self.stack_size = stack_size
        self.frame_stack = deque(maxlen = stack_size)

    def create_stack(self,initial_frame):
        '''
        To be implemented in the begining of a new episode.
        Creates a full stack of the initial frame only and stacks them
        to create a state, which is expanded due to formatting issues..

        Output hace the correct dimensions to be fed into the NN.
        '''
        for i in range(self.stack_size):
            self.frame_stack.append(initial_frame)
        initial_state = np.stack((initial_frame,)*self.stack_size, axis=-1)
        initial_state = np.expand_dims(initial_state, axis=0)
        return initial_state

    def get_new_state(self,new_frame):
        '''
        Add a new state at the end of the stack.

        Output has the correct dimensions to be fed into the NN.
        '''
        self.frame_stack.append(new_frame)
        new_state = np.stack((elem for elem in self.frame_stack),axis=-1)
        new_state = np.expand_dims(new_state, axis=0)
        return new_state
