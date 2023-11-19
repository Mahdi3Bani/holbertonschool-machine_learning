#!/usr/bin/env python3
"""performs K-means on a dataset"""
import numpy as np
import sklearn.cluster


def kmeans(X, k):

    kmeans = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_

    return centroids, labels
