#!/usr/bin/env python3
"""performs K-means on a dataset"""
import sklearn.cluster


def kmeans(X, k):
    """performs K-means on a dataset"""

    kmeans = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_

    return centroids, labels
