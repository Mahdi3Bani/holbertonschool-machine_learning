#!/usr/bin/env python3
"""performs forward propagation over
a pooling layer of a neural network"""


import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    performs forward propagation over
    a pooling layer of a neural network

    A_prev: is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
    containing the output of the previous layer
        *m is the number of examples
        *h_prev is the height of the previous layer
        *w_prev is the width of the previous layer
        *c_prev is the number of channels in the previous layer

    kernel_shape: is a tuple of (kh, kw) containing the size of the
    kernel for the pooling
        *kh is the kernel height
        *kw is the kernel width

    stride: is a tuple of (sh, sw) containing
    the strides for the convolution
        *sh is the stride for the height
        *sw is the stride for the width

    mode: is a string containing either max or avg,
    indicating whether to perform maximum or average pooling,
    respectively


    Returns: the output of the convolutional layer

    """

    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    oh = int(((h_prev - kh) / sh) + 1)
    ow = int(((w_prev - kw) / sw) + 1)

    output = np.zeros(shape=(m, oh, ow, c_prev))

    for h in range(oh):
        for w in range(ow):
            if mode == 'max':
                output[:, h, w, :] = np.max(
                    A_prev[:, h * sh: h * sh + kh, w * sw: w * sw + kw, :],
                    axis=(1, 2)
                )
            else:
                output[:, h, w, :] = np.mean(
                    A_prev[:, h * sh: h * sh + kh, w * sw: w * sw + kw, :],
                    axis=(1, 2)
                )
    return output
