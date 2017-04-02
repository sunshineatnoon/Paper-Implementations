import numpy as np
from scipy.misc import imread, imresize, imsave
import torch


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def deprocess(img):
    img = img.add_(1).div_(2)

    return img
