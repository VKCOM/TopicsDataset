from random import randrange
from typing import Tuple
import numpy as np
from sklearn.base import BaseEstimator
from scipy.spatial.distance import cdist
from modAL.utils.data import modALinput
# TODO fix returning objects for both unimodal and multimodal cases


def diversity_sampling(
        classifier: BaseEstimator,
        X: modALinput,
        n_instances: int = 1,
        transform=None,
        labeled_pool=None,
        index_pool_size=100,
        **uncertainty_measure_kwargs
) -> Tuple[np.ndarray, modALinput]:

    if transform is not None:
        X = transform(X)
        labeled_pool = transform(labeled_pool)

    idx_pool = np.random.randint(low=0, high=len(X), size=min(index_pool_size, len(X)))
    new_to_old_idx = {i: idx for i, idx in enumerate(idx_pool)}

    cur_pool = labeled_pool if labeled_pool is not None else np.array([X[randrange(len(X))]])

    query_idx = []
    for _ in range(n_instances):
        new_idx = new_to_old_idx[cdist(X[idx_pool], cur_pool).min(axis=1).argmax()]
        query_idx.append(new_idx)
        cur_pool = np.vstack([cur_pool, X[new_idx]])

    return query_idx, np.array([])
