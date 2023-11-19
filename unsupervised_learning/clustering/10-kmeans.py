#!/usr/bin/env python3
"""performs K-means on a dataset"""
import numpy as np
from sklearn.cluster import KMeans


def kmeans(X, k):

    kmeans = KMeans(n_clusters=k).fit(X)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_

    return centroids, labels
