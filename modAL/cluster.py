from typing import Tuple
import numpy as np
from modAL.uncertainty import classifier_entropy, classifier_uncertainty, classifier_margin

from modAL.utils.data import modALinput
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
# TODO fix returning objects for both unimodal and multimodal cases

def cluster_sampling(classifier: BaseEstimator, X: modALinput,
                     n_instances: int = 1, transform=None,
                     **uncertainty_measure_kwargs) -> Tuple[np.ndarray, modALinput]:
    entropy = classifier_entropy(classifier, X, **uncertainty_measure_kwargs)
    if transform is not None:
        X = transform(X)

    km = KMeans(n_clusters=n_instances)
    km.fit(X)

    batch = []

    for label in range(n_instances):
        idx = np.where(km.labels_ == label)[0]
        max_entropy = 0
        max_i = 0
        for i in idx:
            if entropy[i] > max_entropy:
                max_entropy = entropy[i]
                max_i = i
        batch.append(max_i)
    query_idx = np.array(batch)
    # print('query idx shape', query_idx.shape)
    # print('query idx', query_idx)
    return query_idx, np.array([])
    # return query_idx, (X if transform is None else X_orig)[query_idx]


def cluster_margin_sampling(classifier: BaseEstimator, X: modALinput,
                     n_instances: int = 1, transform=None,
                     **uncertainty_measure_kwargs) -> Tuple[np.ndarray, modALinput]:
    margin = classifier_margin(classifier, X, **uncertainty_measure_kwargs)
    # print('entropy shape', entropy.shape)
    if transform is not None:
        # X_orig = X
        X = transform(X)

    km = KMeans(n_clusters=n_instances)
    km.fit(X)

    batch = []

    for label in range(n_instances):
        idx = np.where(km.labels_ == label)[0]
        max_entropy = 0
        max_i = 0
        for i in idx:
            if margin[i] > max_entropy:
                max_entropy = margin[i]
                max_i = i
        batch.append(max_i)
    query_idx = np.array(batch)
    # print('query idx shape', query_idx.shape)
    # print('query idx', query_idx)
    return query_idx, np.array([])