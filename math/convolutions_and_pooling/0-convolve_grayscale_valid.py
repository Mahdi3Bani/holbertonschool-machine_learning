#!/usr/bin/env python3
"""performs a valid convolution on grayscale images"""


import numpy as np


def convolve_grayscale_valid(images, kernel):
    """performs a valid convolution on grayscale images"""
    # Get the shape of the input images and the kernel
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Compute the output shape
    ch = h - kh + 1
    cw = w - kw + 1

    # Initialize the output array
    output = np.zeros((m, ch, cw))

    # Loop over the images and the output dimensions
    for i in range(ch):
        for y in range(cw):
            image = images[:, i:i + kh, y:y + kw]
            output[:, i, y] = np.sum(image * kernel, axis=(1, 2))

    return output
