""" Module for the evaluation of feature importance (real-valued and binary) explanations. """

import numpy as np
import sklearn.metrics as metrics

from scipy.spatial.distance import cdist


def cosine_similarity(u, v, bounding: str = 'abs') -> float:
    """
    Computes cosine similarity between two real valued arrays. If negative, returns 0.
     
    :param u: array-like, first real valued vector of dimension n.
    :param v: array-like, second real valued vector of dimension n.
    :param bounding: if the CS is < 0, bound it in [0, 1] via absolute value ('abs') or max(0, value) ('max')
    :return: (0, 1) cosine similarity.
    """
    dist = cdist(u, v, metric='cosine')[0][0]
    if bounding == 'abs':
        return np.abs(dist)
    elif bounding == 'max':
        return max(0, dist)
    else:
        raise ValueError('bounding method not valid.')


def precision(gt, u, **kwargs) -> float:
    """ Computes precision score of two binary arrays
    :param gt: array-like, ground truth labels
    :param u: array-like, predicted labels
    :param kwargs: extra arguments for sklearn.metrics.precision_score
    :return: precision score """
    return metrics.precision_score(gt, u, **kwargs)


def recall(gt, u, **kwargs) -> float:
    """ Computes recall score of two binary arrays
    :param gt: array-like, ground truth labels
    :param u: array-like, predicted labels
    :param kwargs: extra arguments for sklearn.metrics.recall_score
    :return: recall score """
    return metrics.recall_score(gt, u, **kwargs)


def f_score(gt, u, beta=1, **kwargs) -> float:
    """ Computes f-beta score of two binary arrays
        :param gt: array-like, ground truth labels
        :param u: array-like, predicted labels
        :param beta: weight for the score
        :param kwargs: extra arguments for sklearn.metrics.fbeta_score
        :return: f-beta score """
    return metrics.fbeta_score(gt, u, beta=beta, **kwargs)


def auc_score(gt, u, **kwargs) -> float:
    """ Computes roc_auc score of two real valued arrays
        :param gt: array-like, ground truth scores
        :param u: array-like, predicted scores
        :param kwargs: extra arguments for sklearn.metrics.roc_auc_score
        :return: roc_auc score """
    return metrics.roc_auc_score(gt, u, **kwargs)


if __name__ == '__main__':
    pass
