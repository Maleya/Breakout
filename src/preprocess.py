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
