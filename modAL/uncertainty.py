"""
Uncertainty measures and uncertainty based sampling strategies for the active learning models.
"""
from typing import Tuple

import numpy as np
from scipy.stats import entropy
from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator

from modAL.utils.data import modALinput
from modAL.utils.selection import multi_argmax, shuffled_argmax


def _proba_uncertainty(proba: np.ndarray) -> np.ndarray:
    """
    Calculates the uncertainty of the prediction probabilities.

    Args:
        proba: Prediction probabilities.

    Returns:
        Uncertainty of the prediction probabilities.
    """

    return 1 - np.max(proba, axis=1)


def _proba_margin(proba: np.ndarray) -> np.ndarray:
    """
    Calculates the margin of the prediction probabilities.

    Args:
        proba: Prediction probabilities.

    Returns:
        Margin of the prediction probabilities.
    """

    if proba.shape[1] == 1:
        return np.zeros(shape=len(proba))

    part = np.partition(-proba, 1, axis=1)
    margin = - part[:, 0] + part[:, 1]

    return margin


def _proba_entropy(proba: np.ndarray) -> np.ndarray:
    """
    Calculates the entropy of the prediction probabilities.

    Args:
        proba: Prediction probabilities.

    Returns:
        Uncertainty of the prediction probabilities.
    """

    return np.transpose(entropy(np.transpose(proba)))


def classifier_uncertainty(classifier: BaseEstimator, X: modALinput, proba=True, **predict_kwargs) -> np.ndarray:
    """
    Classification uncertainty of the classifier for the provided samples.

    Args:
        classifier: The classifier for which the uncertainty is to be measured.
        X: The samples for which the uncertainty of classification is to be measured.
        **predict_proba_kwargs: Keyword arguments to be passed for the :meth:`predict_proba` of the classifier.

    Returns:
        Classifier uncertainty, which is 1 - P(prediction is correct).
    """
    # calculate uncertainty for each point provided
    try:
        if proba:
            classwise_uncertainty = classifier.predict_proba(X, **predict_kwargs)
        else:
            classwise_uncertainty = classifier.predict(X, **predict_kwargs)
    except NotFittedError:
        return np.ones(shape=(X.shape[0], ))

    # for each point, select the maximum uncertainty
    uncertainty = 1 - np.max(classwise_uncertainty, axis=1)
    return uncertainty


def classifier_margin(classifier: BaseEstimator, X: modALinput, proba=True, **predict_kwargs) -> np.ndarray:
    """
    Classification margin uncertainty of the classifier for the provided samples. This uncertainty measure takes the
    first and second most likely predictions and takes the difference of their probabilities, which is the margin.

    Args:
        classifier: The classifier for which the prediction margin is to be measured.
        X: The samples for which the prediction margin of classification is to be measured.
        **predict_proba_kwargs: Keyword arguments to be passed for the :meth:`predict_proba` of the classifier.

    Returns:
        Margin uncertainty, which is the difference of the probabilities of first and second most likely predictions.
    """
    try:
        # classwise_uncertainty = classifier.predict_proba(X, **predict_proba_kwargs)
        if proba:
            classwise_uncertainty = classifier.predict_proba(X, **predict_kwargs)
        else:
            classwise_uncertainty = classifier.predict(X, **predict_kwargs)
    except NotFittedError:
        return np.zeros(shape=(X.shape[0], ))

    if classwise_uncertainty.shape[1] == 1:
        return np.zeros(shape=(classwise_uncertainty.shape[0],))

    part = np.partition(-classwise_uncertainty, 1, axis=1)
    margin = - part[:, 0] + part[:, 1]

    return margin


def classifier_entropy(classifier: BaseEstimator, X: modALinput, proba=True, **predict_kwargs) -> np.ndarray:
    """
    Entropy of predictions of the for the provided samples.

    Args:
        classifier: The classifier for which the prediction entropy is to be measured.
        X: The samples for which the prediction entropy is to be measured.
        **predict_proba_kwargs: Keyword arguments to be passed for the :meth:`predict_proba` of the classifier.

    Returns:
        Entropy of the class probabilities.
    """
    try:
        if proba:
            classwise_uncertainty = classifier.predict_proba(X, **predict_kwargs)
        else:
            classwise_uncertainty = classifier.predict(X, **predict_kwargs)
    except NotFittedError:
        return np.zeros(shape=(X.shape[0], ))

    return np.transpose(entropy(np.transpose(classwise_uncertainty)))


def classifier_top_entropy(classifier: BaseEstimator, X: modALinput, n_top: int, proba=True, **predict_kwargs) -> np.ndarray:
    """
    Entropy of top n predictions of the for the provided samples.

    Args:
        classifier: The classifier for which the prediction entropy is to be measured.
        X: The samples for which the prediction entropy is to be measured.
        n_top: number of top classes to calculate entropy for
        proba: use predict_proba or juct predict
        **predict_proba_kwargs: Keyword arguments to be passed for the :meth:`predict_proba` of the classifier.

    Returns:
        Entropy of the class probabilities.
    """
    try:
        if proba:
            classwise_uncertainty = classifier.predict_proba(X, **predict_kwargs)
        else:
            classwise_uncertainty = classifier.predict(X, **predict_kwargs)
    except NotFittedError:
        return np.zeros(shape=(X.shape[0], ))

    classwise_uncertainty.sort(axis=1)
    classwise_uncertainty_top = classwise_uncertainty[:, -n_top:]
    return np.transpose(entropy(np.transpose(classwise_uncertainty_top)))


def uncertainty_sampling(classifier: BaseEstimator, X: modALinput,
                         n_instances: int = 1, random_tie_break: bool = False,
                         **uncertainty_measure_kwargs) -> Tuple[np.ndarray, modALinput]:
    """
    Uncertainty sampling query strategy. Selects the least sure instances for labelling.

    Args:
        classifier: The classifier for which the labels are to be queried.
        X: The pool of samples to query from.
        n_instances: Number of samples to be queried.
        random_tie_break: If True, shuffles utility scores to randomize the order. This
            can be used to break the tie when the highest utility score is not unique.
        **uncertainty_measure_kwargs: Keyword arguments to be passed for the uncertainty
            measure function.

    Returns:
        The indices of the instances from X chosen to be labelled;
        the instances from X chosen to be labelled.
    """
    uncertainty = classifier_uncertainty(classifier, X, **uncertainty_measure_kwargs)

    if not random_tie_break:
        query_idx = multi_argmax(uncertainty, n_instances=n_instances)
    else:
        query_idx = shuffled_argmax(uncertainty, n_instances=n_instances)

    if isinstance(X, list) and isinstance(X[0], np.ndarray):
        return query_idx, [x[query_idx] for x in X]
    return query_idx, X[query_idx]


def margin_sampling(classifier: BaseEstimator, X: modALinput,
                    n_instances: int = 1, random_tie_break: bool = False,
                    **uncertainty_measure_kwargs) -> Tuple[np.ndarray, modALinput]:
    """
    Margin sampling query strategy. Selects the instances where the difference between
    the first most likely and second most likely classes are the smallest.
    Args:
        classifier: The classifier for which the labels are to be queried.
        X: The pool of samples to query from.
        n_instances: Number of samples to be queried.
        random_tie_break: If True, shuffles utility scores to randomize the order. This
            can be used to break the tie when the highest utility score is not unique.
        **uncertainty_measure_kwargs: Keyword arguments to be passed for the uncertainty
            measure function.
    Returns:
        The indices of the instances from X chosen to be labelled;
        the instances from X chosen to be labelled.
    """
    margin = classifier_margin(classifier, X, **uncertainty_measure_kwargs)

    if not random_tie_break:
        query_idx = multi_argmax(-margin, n_instances=n_instances)
    else:
        query_idx = shuffled_argmax(-margin, n_instances=n_instances)

    if isinstance(X, list) and isinstance(X[0], np.ndarray):
        return query_idx, [x[query_idx] for x in X]
    return query_idx, X[query_idx]


def margin_trident_sampling(classifier: BaseEstimator, X: modALinput,
                    n_instances: int = 1, random_tie_break: bool = False,
                    combination=(lambda xs: sum(xs)),
                    **uncertainty_measure_kwargs) -> Tuple[np.ndarray, modALinput]:
    # implies trident classifier
    predictions = classifier.predict_proba(X)

    margins = [_proba_margin(prediction) for prediction in predictions]
    # print('margins:', margins[0][:4], margins[1][:4], margins[2][:4])
    margins_combined = [combination([mc, mi, mt]) for mc, mi, mt in zip(margins[0], margins[1], margins[2])]
    # print('margins combined:', margins_combined[:4])
    margins_combined = np.array(margins_combined)

    if not random_tie_break:
        query_idx = multi_argmax(-margins_combined, n_instances=n_instances)
    else:
        query_idx = shuffled_argmax(-margins_combined, n_instances=n_instances)

    if isinstance(X, list) and isinstance(X[0], np.ndarray):
        return query_idx, [x[query_idx] for x in X]
    return query_idx, X[query_idx]


def margin_mixed_sampling(classifier: BaseEstimator, X: modALinput,
                    n_instances: int = 1, random_tie_break: bool = False,
                    **uncertainty_measure_kwargs) -> Tuple[np.ndarray, modALinput]:
    # implies trident classifier
    predictions = classifier.predict_proba(X)

    margins = [_proba_margin(prediction) for prediction in predictions]
    # print('margins:', margins[0][:4], margins[1][:4], margins[2][:4])
    instance_per_modality = n_instances // len(predictions) + 1

    if not random_tie_break:
        query_idx_1 = multi_argmax(-margins[0], n_instances=instance_per_modality)
        query_idx_2 = multi_argmax(-margins[1], n_instances=instance_per_modality)
        query_idx_3 = multi_argmax(-margins[2], n_instances=instance_per_modality)
        query_idx = np.concatenate([query_idx_1, query_idx_2, query_idx_3], axis=0)[:n_instances]
    else:
        query_idx_1 = shuffled_argmax(-margins[0], n_instances=instance_per_modality)
        query_idx_2 = shuffled_argmax(-margins[1], n_instances=instance_per_modality)
        query_idx_3 = shuffled_argmax(-margins[2], n_instances=instance_per_modality)
        query_idx = np.concatenate([query_idx_1, query_idx_2, query_idx_3], axis=0)[:n_instances]


    if isinstance(X, list) and isinstance(X[0], np.ndarray):
        return query_idx, [x[query_idx] for x in X]
    return query_idx, X[query_idx]


def margin_multitask_sampling(classifier: BaseEstimator, X: modALinput,
                    n_instances: int = 1, random_tie_break: bool = False,
                    **uncertainty_measure_kwargs) -> Tuple[np.ndarray, modALinput]:
    # implies trident classifier
    predictions = classifier.predict_proba(X)

    margins = [_proba_margin(prediction) for prediction in predictions]
    print(margins[0][:10], margins[1][:10])
    margins_combined = [mt * mw for mt, mw in zip(margins[0], margins[1])]
    print(margins_combined[:10])
    margins_combined = np.array(margins_combined)
    print(margins_combined[:10])

    if not random_tie_break:
        query_idx = multi_argmax(-margins_combined, n_instances=n_instances)
    else:
        query_idx = shuffled_argmax(-margins_combined, n_instances=n_instances)

    if isinstance(X, list) and isinstance(X[0], np.ndarray):
        return query_idx, [x[query_idx] for x in X]
    return query_idx, X[query_idx]


def entropy_sampling(classifier: BaseEstimator, X: modALinput,
                     n_instances: int = 1, random_tie_break: bool = False,
                     **uncertainty_measure_kwargs) -> Tuple[np.ndarray, modALinput]:
    """
    Entropy sampling query strategy. Selects the instances where the class probabilities
    have the largest entropy.

    Args:
        classifier: The classifier for which the labels are to be queried.
        X: The pool of samples to query from.
        n_instances: Number of samples to be queried.
        random_tie_break: If True, shuffles utility scores to randomize the order. This
            can be used to break the tie when the highest utility score is not unique.
        **uncertainty_measure_kwargs: Keyword arguments to be passed for the uncertainty
            measure function.

    Returns:
        The indices of the instances from X chosen to be labelled;
        the instances from X chosen to be labelled.
    """
    entropy = classifier_entropy(classifier, X, **uncertainty_measure_kwargs)

    if not random_tie_break:
        query_idx = multi_argmax(entropy, n_instances=n_instances)
    else:
        query_idx = shuffled_argmax(entropy, n_instances=n_instances)

    if isinstance(X, list) and isinstance(X[0], np.ndarray):
        return query_idx, [x[query_idx] for x in X]
    return query_idx, X[query_idx]


def entropy_top_sampling(classifier: BaseEstimator, X: modALinput,
                       n_instances: int = 1, random_tie_break: bool = False,
                       n_top: int = 5,
                       **uncertainty_measure_kwargs) -> Tuple[np.ndarray, modALinput]:
    """
    Entropy sampling query strategy. Selects the instances where the class probabilities
    have the largest entropy.

    Args:
        classifier: The classifier for which the labels are to be queried.
        X: The pool of samples to query from.
        n_instances: Number of samples to be queried.
        random_tie_break: If True, shuffles utility scores to randomize the order. This
            can be used to break the tie when the highest utility score is not unique.
        n_top: number of top classes to calculate entropy
        **uncertainty_measure_kwargs: Keyword arguments to be passed for the uncertainty
            measure function.

    Returns:
        The indices of the instances from X chosen to be labelled;
        the instances from X chosen to be labelled.
    """
    entropy = classifier_top_entropy(classifier, X, n_top, **uncertainty_measure_kwargs)

    if not random_tie_break:
        query_idx = multi_argmax(entropy, n_instances=n_instances)
    else:
        query_idx = shuffled_argmax(entropy, n_instances=n_instances)

    if isinstance(X, list) and isinstance(X[0], np.ndarray):
        return query_idx, [x[query_idx] for x in X]
    return query_idx, X[query_idx]
