from typing import Union, Tuple

import numpy as np
import scipy.sparse as sp

from modAL.models.base import BaseLearner, BaseCommittee


def passive_strategy(classifier: Union[BaseLearner, BaseCommittee],
                       X: Union[np.ndarray, sp.csr_matrix],
                       n_instances: int = 20,
                        **trash
                       ) -> Tuple[Union[int, np.ndarray], list]:
    if isinstance(X, list) and isinstance(X[0], np.ndarray):
        x_size = X[0].shape[0]
    else:
        x_size = X.shape[0]

    query_idx = np.random.randint(low=0, high=x_size, size=n_instances)
    if isinstance(X, list) and isinstance(X[0], np.ndarray):
        return query_idx, [x[query_idx] for x in X]
    return query_idx, X[query_idx]
