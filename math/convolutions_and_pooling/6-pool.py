#!/usr/bin/env python3
"""that performs a convolution on grayscale images"""


import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """that performs a convolution on grayscale images"""
    # Get the shape of the input images and the kernel
    m, h, w, c = images.shape
    kh, kw = kernel_shape

    sh, sw = stride

    # Compute the output shape
    oh = int((h - kh) / sh + 1)
    ow = int((w - kw) / sw + 1)

    # Initialize the output array
    output = np.zeros(shape=(m, oh, ow, c))

    for x in range(oh):
        for y in range(ow):
            if mode == "max":
                output[:, x, y, :] = images[:, x * sh:x * sh +
                                            kh, y * sw:y * sw + kw, :].max(axis=(1, 2))
            else:
                output[:, x, y, :] = np.mean(
                    images[:, x * sh:x * sh + kh, y * sw:y * sw + kw, :],
                    axis=(1, 2)
                )

    return output
