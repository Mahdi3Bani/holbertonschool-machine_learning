#!/usr/bin/env python3
"""performs a same convolution on grayscale images"""


import numpy as np


def convolve_grayscale_same(images, kernel):
    """performs a same convolution on grayscale images"""
    # Get the shape of the input images and the kernel
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Compute the output shape
    ph = int((kh - 1) / 2) if kh % 2 != 0 else int(kh / 2)
    pw = int((kw - 1) / 2) if kw % 2 != 0 else int(kw / 2)

    # Initialize the output array
    output = np.zeros((m, h, w))

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

    for x in range(h):
        for y in range(w):
            output[:, x, y] = np.sum(
                (kernel * images_padded[:, x:x + kh, y:y + kw]),
                axis=(1, 2)
            )

    return output
