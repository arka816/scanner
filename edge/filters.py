import numpy as np
from scipy.signal import convolve2d

PREWITT_KERNEL_X = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
PREWITT_KERNEL_Y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]) 

SOBEL_KERNEL_X   = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
SOBEL_KERNEL_Y   = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

def gaussian_kernel(kernel_size, sd):
    # generates a gaussian kernel
    ax = np.linspace(-(kernel_size - 1) / 2, (kernel_size - 1) / 2, kernel_size)
    gauss_1d = np.exp(-0.5 * np.square(ax) / np.square(sd))
    gauss_2d = np.outer(gauss_1d, gauss_1d)
    return gauss_2d / np.sum(gauss_2d)

def gaussian(image, kernel_size, sd=1):
    kernel = gaussian_kernel(kernel_size, sd)
    return convolve2d(image, kernel, mode='same', boundary='symm')

def prewitt(image):
    Gx = convolve2d(image, PREWITT_KERNEL_X, mode='same', boundary='symm')
    Gy = convolve2d(image, PREWITT_KERNEL_Y, mode='same', boundary='symm')
    return Gx, Gy

def sobel(image):
    Gx = convolve2d(image, SOBEL_KERNEL_X, mode='same', boundary='symm')
    Gy = convolve2d(image, SOBEL_KERNEL_Y, mode='same', boundary='symm')
    return Gx, Gy
