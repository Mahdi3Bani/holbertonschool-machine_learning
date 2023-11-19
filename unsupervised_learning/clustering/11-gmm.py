#!/usr/bin/env python3
"""calculates a GMM from a dataset"""
from sklearn.mixture import GaussianMixture


def gmm(X, k):
    """calculates a GMM from a dataset"""
    gmm = GaussianMixture(n_components=k).fit(X)
    pi = gmm.weights_
    m = gmm.means_
    S = gmm.covariances_
    clss = gmm.predict(X)
    bic = gmm.bic(X)
    return pi, m, S, clss, bic
