from __future__ import print_function
from PIL import Image
import numpy as np
import scipy.misc
from matplotlib import pyplot as plt
import six

def open_and_resize_image(path, target_width):
    image = Image.open(path).convert('RGB')
    width, height = image.size
    target_height = int(round(float(height * target_width) / width))
    image = image.resize((target_width, target_height), Image.BILINEAR)
    return image

def match_color_histogram(x, y):
    z = np.zeros_like(x)
    shape = x[0].shape
    for i in six.moves.range(len(x)):
        a = x[i].reshape((3, -1))
        a_mean = np.mean(a, axis=1, keepdims=True)
        a_var = np.cov(a)
        d, v = np.linalg.eig(a_var)
        d += 1e-6
        a_sigma_inv = v.dot(np.diag(d ** (-0.5))).dot(v.T)

        b = y[i].reshape((3, -1))
        b_mean = np.mean(b, axis=1, keepdims=True)
        b_var = np.cov(b)
        d, v = np.linalg.eig(b_var)
        b_sigma = v.dot(np.diag(d ** 0.5)).dot(v.T)

        transform = b_sigma.dot(a_sigma_inv)
        z[i,:] = (transform.dot(a - a_mean) + b_mean).reshape(shape)
    return z
'''
def match_color_histogram(x, y):
        shape = x[0].shape
        x = x[0].reshape((3,-1)) # 3 x (256x256)I
        y = y[0].reshape((3,-1))

        x_mean = np.mean(x, axis=1, keepdims=True)
        y_mean = np.mean(y, axis=1, keepdims=True)
        x_var = np.cov(x)
        y_var = np.cov(y)

        Ls = np.linalg.cholesky(x_var)
        Lc = np.linalg.cholesky(y_var)

        A = np.dot(Lc,np.linalg.inv(Ls))
        b = y_mean - np.dot(A,x_mean)

        S_ = np.dot(x.transpose(), A).transpose() + b
        S_ = np.expand_dims(S_.reshape(shape),0)
        return S_.astype('float32') 
'''
