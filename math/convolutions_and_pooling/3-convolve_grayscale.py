#!/usr/bin/env python3
"""that performs a convolution on grayscale images"""


import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """that performs a convolution on grayscale images"""
    # Get the shape of the input images and the kernel
    m, h, w = images.shape
    kh, kw = kernel.shape


    sh, sw = stride

    if padding == "valid":
        ph, pw = 0, 0
    elif padding == "same":
        ph = int((((h - 1) * sh + kh - h) / 2) + 1)
        pw = int((((w - 1) * sw + kw - w) / 2) + 1)
    
    else:
        ph, pw= padding
    # Compute the output shape
    oh = int((h - kh + 2 * ph) / sh + 1)
    ow = int((w - kw + 2 * pw) / sw + 1)

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
