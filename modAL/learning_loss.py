from typing import Union, Tuple

import numpy as np

import torch
import torch.nn.functional as F

from modAL.utils import multi_argmax

from modAL.models.base import BaseLearner, BaseCommittee

import scipy.sparse as sp
from scipy.stats import entropy


def learning_loss_strategy(classifier: Union[BaseLearner, BaseCommittee],
                           X: Union[np.ndarray, sp.csr_matrix],
                           n_instances: int = 20,
                           **predict_kwargs
                           ) -> Tuple[np.ndarray, Union[np.ndarray, sp.csr_matrix]]:
    losses = classifier.estimator.predict_loss(X, **predict_kwargs)
    query_idx = multi_argmax(losses, n_instances=n_instances).squeeze(axis=1)
    return query_idx, np.array([])


def learning_loss_ideal(classifier: Union[BaseLearner, BaseCommittee],
                        X: Union[np.ndarray, sp.csr_matrix],
                        y,
                        n_instances: int = 20,
                        is_reverse=False,
                        **predict_kwargs
                        ) -> Tuple[np.ndarray, Union[np.ndarray, sp.csr_matrix]]:
    prediction = torch.tensor(classifier.estimator.predict(X))
    actual = torch.argmax(torch.tensor(y), dim=1)
    losses = F.nll_loss(prediction, actual, reduction='none').detach().numpy()
    metric = losses if not is_reverse else 1 - losses
    query_idx = multi_argmax(metric, n_instances=n_instances)

    return query_idx, np.array([])


def get_least_confidence(predictions):
    return 1 - np.max(predictions, axis=1)


def get_margin(predictions):
    part = np.partition(-predictions, 1, axis=1)
    margin = - part[:, 0] + part[:, 1]
    return -margin


def get_entropy(predictions):
    return np.transpose(entropy(np.transpose(predictions)))


uncertainty_measure_dict = {
    'least_confident': get_least_confidence,
    'margin': get_margin,
    'entropy': get_entropy
}


def learning_loss_ideal_with_uncertainty(classifier: Union[BaseLearner, BaseCommittee],
                        X: Union[np.ndarray, sp.csr_matrix],
                        y,
                        n_instances: int = 20,
                        is_reverse=False,
                        uncertainty_measure='entropy',
                        use_more_uncertain=True,
                        **predict_kwargs
                        ) -> Tuple[np.ndarray, Union[np.ndarray, sp.csr_matrix]]:
    if uncertainty_measure not in uncertainty_measure_dict:
        raise ValueError('uncertainty measure can be equal only to "least_confident", "margin" or "entropy"')

    prediction = torch.tensor(classifier.estimator.predict(X))

    uncertainty = uncertainty_measure_dict[uncertainty_measure](prediction.detach().numpy())
    median_uncertainty = np.median(uncertainty)
    uncertain_enough = uncertainty >= median_uncertainty

    actual = torch.argmax(torch.tensor(y), dim=1)
    losses = F.nll_loss(prediction, actual, reduction='none').detach().numpy()
    losses += min(losses)

    metric = losses if not is_reverse else 1 - losses

    filtered_metric = 1 * uncertain_enough * metric
    query_idx = multi_argmax(filtered_metric, n_instances=n_instances)

    return query_idx, np.array([])
