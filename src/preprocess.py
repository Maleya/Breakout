"""
Preprocessing function that expects an np.array with shape (210, 160, 3)
"""

from PIL import Image
import numpy as np


def preprocess(im):
    im = Image.fromarray(im)
    im = im.convert('L')
    im = im.crop((8, 31, 152, 210))  # somewhat good settings for breakout
    im = im.resize((84, 84))
    # im.show()
    return(np.array(im))


if __name__ == "__main__":
    # generate a realistic frame
    import gym
    env = gym.make('BreakoutDeterministic-v4')
    frame = env.reset()
    frame, reward, done, _, = env.step(env.action_space.sample())
    env.close()
    output = preprocess(frame)
    print(output)
    print(f"input shape: {frame.shape}, output shape: {output.shape}")
