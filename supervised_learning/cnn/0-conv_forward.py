#!/usr/bin/env python3
"""performs forward propagation over a
convolutional layer of a neural network"""


import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    performs forward propagation over a convolutional layer of a neural network

    A_prev: is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
    containing the output of the previous layer
        *m is the number of examples
        *h_prev is the height of the previous layer
        *w_prev is the width of the previous layer
        *c_prev is the number of channels in the previous layer

    W: is a numpy.ndarray of shape (kh, kw, c_prev, c_new)
    containing the kernels for the convolution
        *kh is the filter height
        *kw is the filter width
        *c_prev is the number of channels in the previous layer
        *c_new is the number of channels in the output

    b:is a numpy.ndarray of shape (1, 1, 1, c_new)
    containing the biases applied to the convolution

    activation: is an activation function applied to the convolution

    padding: is a string that is either same or valid,
    indicating the type of padding used

    stride: is a tuple of (sh, sw) containing
    the strides for the convolution
        *sh is the stride for the height
        *sw is the stride for the width

    Returns: the output of the convolutional layer

    """

    m, h_prev, w_prev, _ = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    if padding == 'valid':
        ph = pw = 0
    else:
        ph = int(((h_prev - 1) * sh + kh - h_prev) // 2)
        pw = int(((w_prev - 1) * sw + kw - w_prev) // 2)
    # output height
    oh = int(((2 * ph - kh + h_prev) / sh) + 1)
    # output weight
    ow = int(((2 * pw - kw + w_prev) / sw) + 1)

    output = np.zeros(shape=(m, oh, ow, c_new))

    A_prev = np.pad(A_prev,
                    pad_width=(
                        (0, 0),
                        (ph, ph),
                        (pw, pw),
                        (0, 0)
                    ),
                    mode='constant')

    for h in range(oh):
        for w in range(ow):
            for ch in range(c_new):
                output[:, h, w, ch] = np.sum(
                    A_prev[:, h * sh: h * sh + kh, w * sw: w * sw + kw, :]
                    * W[:, :, :, ch], axis=(1, 2, 3)
                )
    return activation(output + b)
