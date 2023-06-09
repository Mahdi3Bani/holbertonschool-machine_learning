#!/usr/bin/env python3
"""that performs back propagation over a pooling
    layer of a neural network"""


import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    that performs back propagation over a pooling
    layer of a neural network

    dA is a numpy.ndarray of shape (m, h_new, w_new, c_new)
    containing the partial derivatives with respect to the
    output of the pooling layer
        *m is the number of examples
        *h_new is the height of the output
        *w_new is the width of the output
        *c is the number of channels



    A_prev: is a numpy.ndarray of shape (m, h_prev, w_prev, c)
    containing the output of the previous layer
        *h_prev is the height of the previous layer
        *w_prev is the width of the previous layer



    kernel_shape is a tuple of (kh, kw) containing the
    size of the kernel for the pooling
        *kh is the kernel height
        *kw is the kernel width


    stride is a tuple of (sh, sw) containing the strides for the pooling
        *sh is the stride for the height
        *sw is the stride for the width

    mode is a string containing either max or avg, indicating whether
    to perform maximum or average pooling, respectively

    Returns: the partial derivatives with respect to the
    previous layer (dA_prev)



    """
    m, h_prev, w_prev, _ = A_prev.shape
    _, h_new, w_new, c = dA.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros(A_prev.shape)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for ch in range(c):
                    if mode == 'max':
                        mask = A_prev[i,
                                      h * sh: h * sh + kh,
                                      w * sw: w * sw + kw,
                                      ch] == np.max(
                            A_prev[i,
                                   h * sh: h * sh + kh,
                                   w * sw: w * sw + kw,
                                   ch]
                        )
                        dA_prev[i,
                                h * sh: h * sh + kh,
                                w * sw: w * sw + kw,
                                ch] += mask * dA[i, h, w, ch]
                    else:
                        dA_prev[i,
                                h * sh: h * sh + kh,
                                w * sw: w * sw + kw,
                                ch] += np.ones((kh, kw)) * (
                            dA[i, h, w, ch] / (kh * kw)
                        )
    return dA_prev
