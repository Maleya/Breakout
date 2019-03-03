"""
Preprocessing function that expects an np.array with shape (210, 160, 3)
"""

from PIL import Image
import numpy as np


def pre_process_BO(im):
    im = Image.fromarray(im)
    im = im.convert('L')
    im = im.crop((0, 31, 160, 210))  # somewhat good settings for breakout
    im = im.resize((84, 84))
    return(np.array(im))
    # im.show()


if __name__ == "__main__":
    # generate a realistic frame
    import gym
    env = gym.make('Breakout-v0')
    frame = env.reset()
    frame, reward, done, _, = env.step(env.action_space.sample())
    env.close()
    output = pre_process_BO(frame)
    print(output)
    print(f"input shape: {frame.shape}, output shape: {output.shape}")
