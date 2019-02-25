"""
Preprocessing function that expects an np.array with shape (210, 160, 3)
greyscales, crops and downsamples the image ending in (84, 76)
"""

import numpy as np

# ---- test code needed to generate a frame ---------
# import gym
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# env = gym.make('MsPacman-v0')


# frame = env.reset()
# done = False
# # while done == False:
# frame, reward, done, _, = env.step(env.action_space.sample())
# env.close()
# print(frame.shape)
# ------ test code end ------------------------------


def pre_process(img):
    img = grayscale(img)
    img = crop(img,4,156,2,170) # good pacman values (alt: 0,164,2,170)
    img = downsample(img,2)
    return img
    
def grayscale(img):
    ''' from np.array dim 2 to 1'''
    return np.mean(img,axis=2).astype(np.uint8)

def downsample(img,factor):
    '''Samples down by factor '''
    return img[::factor,::factor]

def crop(img,x_start,x_end,y_start,y_end):
    '''Crop away pixels'''
    return img[y_start:y_end,x_start:x_end]
