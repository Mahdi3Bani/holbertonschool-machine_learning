#!/usr/bin/env python3
"""performs back propagation over a convolutional
layer of a neural network"""


import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    performs back propagation over a convolutional
    layer of a neural network

    dZ: numpy.ndarray of shape (m, h_new, w_new, c_new) containing the partial derivatives with respect to the unactivated output of the convolutional layer
        *m is the number of examples
        *h_new is the height of the output
        *w_new is the width of the output
        *c_new is the number of channels in the output


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

    padding: is a string that is either same or valid,
    indicating the type of padding used

    stride: is a tuple of (sh, sw) containing
    the strides for the convolution
        *sh is the stride for the height
        *sw is the stride for the width

    Returns: the partial derivatives with respect
    to the previous layer (dA_prev), the kernels (dW),
    and the biases (db), respectively


    """
    _, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, _ = A_prev.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride

    if padding == "same":
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    else:
        ph, pw = 0, 0

    A_prev_pad = np.pad(
        A_prev,
        [(0, 0), (ph, ph), (pw, pw), (0, 0)],
        mode="constant"
    )
    dA_padded = np.zeros(shape=A_prev_pad.shape)
    dW = np.zeros(shape=W.shape)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    hs = h * sh
                    ws = w * sw
                    dA_padded[i,
                              hs:hs + kh,
                              ws:ws + kw,
                              :] += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += A_prev_pad[i,
                                                     hs:hs + kh,
                                                     ws:ws + kw,
                                                     :] * dZ[i, h,
                                                             w, c]

    if padding == "same":
        dA = dA_padded[:, ph:-ph, pw:-pw, :]
    else:
        dA = dA_padded

    return dA, dW, db
