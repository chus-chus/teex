""" Module for the evaluation of feature importance (real-valued and binary) explanations. """

import numpy as np
import sklearn.metrics as met

from scipy.spatial.distance import cdist


# noinspection PyUnboundLocalVariable
def feature_importance_scores(gt, u, metrics=None, binarizeExp=False, **kwargs):
    """ Computes fscore, precision, recall or / and cosine similarity between two feature importance vectors.

    :param gt: (array-like) ground truth feature importance vector.
    :param u: (array-like) predicted feature importance vector.
    :param metrics: (str / array-like) metric/s to be computed: 'fscore', 'prec', 'rec', 'cs', 'auc'
    :param binarizeExp: (bool) should the vectors be binarized for computing fscore, precision and recall?
    :return: (list) computed metrics.
    """
    if metrics is None:
        metrics = ['fscore']
    elif isinstance(metrics, str):
        metrics = [metrics]

    if binarizeExp:
        binaryGt = [1 if item > 0 else 0 for item in gt]
        binaryU = [1 if item > 0 else 0 for item in u]

    ret = []
    for metric in metrics:
        if metric == 'fscore':
            if binarizeExp:
                ret.append(f_score(binaryGt, binaryU, **kwargs))
            else:
                ret.append(f_score(gt, u, **kwargs))
        elif metric == 'prec':
            if binarizeExp:
                ret.append(precision(binaryGt, binaryU, **kwargs))
            else:
                ret.append(precision(gt, u, **kwargs))
        elif metric == 'rec':
            if binarizeExp:
                ret.append(recall(binaryGt, binaryU, **kwargs))
            else:
                ret.append(recall(gt, u, **kwargs))
        elif metric == 'cs':
            ret.append(cosine_similarity(gt, u, **kwargs))
        elif metric == 'auc':
            ret.append(auc_score(gt, u, **kwargs))
        else:
            raise ValueError(f"Invalid metric. Use {['fscore', 'prec', 'rec', 'cs', 'auc']}")

    return ret


def cosine_similarity(u, v, bounding: str = 'abs') -> float:
    """
    Computes cosine similarity between two real valued arrays. If negative, returns 0.

    :param u: array-like, real valued vector of dimension n.
    :param v: array-like, real valued vector of dimension n.
    :param bounding: if the CS is < 0, bound it in [0, 1] via absolute value ('abs') or max(0, value) ('max')
    :return: (0, 1) cosine similarity.
    """
    dist = 1 - cdist([u], [v], metric='cosine')[0][0]
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
    return met.precision_score(gt, u, **kwargs)


def recall(gt, u, **kwargs) -> float:
    """ Computes recall score of two binary arrays
    :param gt: array-like, ground truth labels
    :param u: array-like, predicted labels
    :param kwargs: extra arguments for sklearn.metrics.recall_score
    :return: recall score """
    return met.recall_score(gt, u, **kwargs)


def f_score(gt, u, beta=1, **kwargs) -> float:
    """ Computes f-beta score of two binary arrays
        :param gt: array-like, ground truth labels
        :param u: array-like, predicted labels
        :param beta: weight for the score
        :param kwargs: extra arguments for sklearn.metrics.fbeta_score
        :return: f-beta score """
    return met.fbeta_score(gt, u, beta=beta, **kwargs)


def auc_score(gt, u, **kwargs) -> float:
    """ Computes roc_auc score of two real valued arrays
        :param gt: array-like, ground truth scores
        :param u: array-like, predicted scores
        :param kwargs: extra arguments for sklearn.metrics.roc_auc_score
        :return: roc_auc score """
    return met.roc_auc_score(gt, u, **kwargs)


if __name__ == '__main__':
    pass
