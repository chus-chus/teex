""" Module for the evaluation of explanations. """

import numpy as np

from scipy.spatial.distance import cdist
from sklearn.metrics import fbeta_score


def feature_importance_similarity(u: np.array, v: np.array) -> float:
    """
    Computes cosine similarity between two real valued vectors. If negative, returns 0.
     
    :param u: array, first real valued vector of dimension n.
    :param v: array, second real valued vector of dimension n.
    :return: (0, 1) cosine similarity.
    """
    return max(0, cdist(u, v, metric='cosine')[0][0])


def pixel_importance_similarity(gt: np.array, image: np.array, metric: str = 'fscore',
                                threshold: float = None, beta: float = 1, **kwargs: dict) -> float:
    """
    Metrics for the evaluation of image pixel importance explanations.

    :param gt: array, ground truth pixel importance binary mask. Non-important segments must be 0.
    :param image: array, image pixel importance explanation.
    :param metric: 'fscore', 'prec' or 'rec'.
    :param threshold: If not None, (0, 1) normalised threshold for computing the score. Values >= threshold will be
                      considered.
    :param beta: beta value for the fscore.
    :return: the specified similarity metric.
    """

    if metric == 'fscore':
        return fbeta_score(gt, image, beta=)

    pass
