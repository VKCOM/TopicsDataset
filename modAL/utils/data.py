from typing import Union, Container
from itertools import chain

import numpy as np
import scipy.sparse as sp


modALinput = Union[list, np.ndarray, sp.csr_matrix]


def data_vstack(blocks: Container) -> modALinput:
    """
    Stack vertically both sparse and dense arrays.

    Args:
        blocks: Sequence of modALinput objects.

    Returns:
        New sequence of vertically stacked elements.
    """
    if isinstance(blocks[0], np.ndarray):
        return np.concatenate(blocks)
    elif isinstance(blocks[0], list):
        if isinstance(blocks[0][0], np.ndarray):
            return [np.concatenate([blocks[i][j] for i in range(len(blocks))]) for j in range(len(blocks[0]))]
        return list(chain(blocks))
    elif sp.issparse(blocks[0]):
        return sp.vstack(blocks)
    else:
        try:
            return np.concatenate(blocks)
        except:
            raise TypeError('%s datatype is not supported' % type(blocks[0]))
