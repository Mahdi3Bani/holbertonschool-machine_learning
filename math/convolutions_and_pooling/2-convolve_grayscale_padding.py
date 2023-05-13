#!/usr/bin/env python3
"""performs a convolution on grayscale images with custom padding"""


import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """performs a convolution on grayscale images with custom padding"""
    # Get the shape of the input images and the kernel
    m, h, w = images.shape
    kh, kw = kernel.shape


    ph,pw = padding
    # Compute the output shape
    oh = h - kh + 2 * ph + 1
    ow = w - kw + 2 * pw + 1

    # Initialize the output array
    output = np.zeros(shape=(m, oh, ow))

    # Loop over the images and the output dimensions
    images_padded = np.pad(
        images,
        [
            (0, 0),
            (ph, ph),
            (pw, pw)
        ],
        mode="constant"
    )

    for x in range(oh):
        for y in range(ow):
            output[:, x, y] = np.sum(
                (kernel * images_padded[:, x:x + kh, y:y + kw]),
                axis=(1, 2)
            )

    return output
