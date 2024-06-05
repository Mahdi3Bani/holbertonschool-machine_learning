#!/usr/bin/env python3
'''comment'''


import numpy as np


class GaussianProcess:
    '''GaussianProcess class'''

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        '''calculates the covariance kernel matrix between two matrices'''

        sq_dist = np.sum(X1 ** 2, 1).reshape(-1, 1) +\
            np.sum(X2 ** 2, 1) - 2 * np.matmul(X1, X2.T)

        return self.sigma_f**2 * np.exp(-0.5 / self.l ** 2 * sq_dist)

    def predict(self, X_s):
        '''predicts the mean and standard deviation of points in a Gaussian process '''

        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(self.K)
        mu_s = K_s.T.dot(K_inv).dot(self.Y).flatten()
        cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
        sigma_s = np.diag(cov_s)

        return mu_s, sigma_s

    def update(self, X_new, Y_new):
        self.X = np.concatenate((self.X, [X_new]))
        self.Y = np.concatenate((self.Y, [Y_new]))
        self.K = self.kernel(self.X, self.X)

