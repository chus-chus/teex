""" Module for the evaluation of feature importance (real-valued and binary) explanations. """

import numpy as np
import sklearn.metrics as met

from scipy.spatial.distance import cdist


_AVAILABLE_FEATURE_IMPORTANCE_METRICS = {'fscore', 'prec', 'rec', 'cs', 'auc'}


def _binarize_vectors(v, threshold):
    """ Binarize 1 or more array-like object/s of floats to have values :math:`\in \{0, 1\}`.
    :param v: (1d array-like or 2d array-like of shape (n_features, n_samples)).
    :return: (1d ndarray of shape (n_features,) or 2d ndarray of shape (n_features, n_samples)). """

    assert 0 < threshold < 1, 'threshold should be a value between 0 and 1.'
    if isinstance(v, np.ndarray):
        return (v > threshold).astype(np.int32)
    elif isinstance(v, (list, tuple)):
        return (np.array(v) > threshold).astype(np.int32)
    else:
        raise ValueError('Format not supported.')


def feature_importance_scores(gts, preds, metrics=None, average=True, binThreshold=0.5):
    # todo move auc to classification score
    """ Computes quality metrics between one or more feature importance vectors.

    :param gts: (1d np.ndarray or 2d np.ndarray of shape (n_features, n_samples)) ground truth feature importance
    vectors.
    :param preds: (1d np.ndarray or 2d np.ndarray of shape (n_features, n_samples)) predicted feature importance
    vectors.
    :param metrics: (str or array-like of str) metric/s to be computed. Available:
        - 'fscore': Computes the F1 Score between the ground truths and the predicted vectors.
        - 'prec': Computes the Precision Score between the ground truths and the predicted vectors.
        - 'rec': Computes the Recall Score between the ground truths and the predicted vectors.
        - 'auc': Computes the ROC AUC Score between the ground truths and the predicted vectors.
        - 'cs': Computes the Cosine Similarity between the ground truths and the predicted vectors.
    The vectors are automatically binarized for computing recall, precision and fscore.
    :param average: (bool) (bool, default :code:`True`) Used only if :code:`gt` and :code:`rule` contain multiple
    observations. Should the computed metrics be averaged across all the samples?
    :param binThreshold: (float in [0, 1]) features with a value bigger than this will be set 1 and 0 otherwise when
    binarizing for the computation of 'fscore', 'prec', 'rec' and 'cs'.
    :return: (ndarray of shape (n_metrics,) or (n_samples, n_metrics)) specified metric/s in the original order. """

    if metrics is None:
        metrics = ['fscore']
    elif isinstance(metrics, str):
        metrics = [metrics]

    for metric in metrics:
        if metric not in _AVAILABLE_FEATURE_IMPORTANCE_METRICS:
            raise ValueError(f"'{metric}' metric not valid. Use {_AVAILABLE_FEATURE_IMPORTANCE_METRICS}")

    if not isinstance(gts, np.ndarray) or not isinstance(preds, np.ndarray):
        raise ValueError('Ground truths and predictions must be np.ndarrays.')

    binaryGts = _binarize_vectors(gts, threshold=binThreshold)
    binaryPreds = _binarize_vectors(preds, threshold=binThreshold)

    if len(binaryPreds.shape) == 1:
        binaryGts, binaryPreds = binaryGts.reshape(1, -1), binaryPreds.reshape(1, -1)
        gts, preds = gts.reshape(1, -1), preds.reshape(1, -1)

    ret = []
    for binGt, binPred, gt, pred in zip(binaryGts, binaryPreds, gts, preds):
        mets = []
        # todo what to do when we have only 1 class
        for metric in metrics:
            if metric == 'fscore':
                mets.append(f_score(binGt, binPred))
            elif metric == 'prec':
                mets.append(precision(binGt, binPred))
            elif metric == 'rec':
                mets.append(recall(binGt, binPred))
            elif metric == 'cs':
                mets.append(cosine_similarity(gt, pred))
            elif metric == 'auc':
                if len(np.unique(binGt)) == 1:
                    mets.append(np.nan)
                else:
                    mets.append(auc_score(binGt, pred))
        ret.append(mets)

    ret = np.array(ret).astype(np.float32)

    if average is True and binaryPreds.shape[0] > 1:
        ret = np.mean(ret, axis=0)
    elif binaryPreds.shape[0] == 1:
        return ret.squeeze()

    return ret


def cosine_similarity(u, v, bounding: str = 'abs') -> float:
    """
    Computes cosine similarity between two real valued arrays. If negative, returns 0.

    :param u: (array-like), real valued vector of dimension n.
    :param v: array-like, real valued vector of dimension n.
    :param bounding: if the CS is < 0, bound it in [0, 1] via absolute value ('abs') or max(0, value) ('max')
    :return: (0, 1) cosine similarity. """

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
