""" Module for evaluation of feature importance explanations. """


import warnings

import numpy as np
import sklearn.metrics as met

from scipy.spatial.distance import cdist

from teex._utils._arrays import _binarize_arrays, _check_correct_array_values
from teex._utils._errors import MetricNotAvailableError
from teex.featureImportance.data import scale_fi_bounds

_AVAILABLE_FEATURE_IMPORTANCE_METRICS = {'fscore', 'prec', 'rec', 'cs', 'auc'}


def _individual_fi_metrics(gt, pred, binGt, binPred, metric, predsNegative, thresholdType):
    """ Classification and real vector metrics. If metric='auc' and predsNegative=True, 'pred' is modified accordingly
    (see :func:`feature_importance_scores`).

     :param gt: (ndarray) of shape (nFeatures,). Ground truth (real or binary) vector.
     :param pred: (ndarray) of shape (nFeatures,). Predicted (real or binary) vector.
     :param binGt: (ndarray) of shape (nFeatures,). Ground truth (binary) vector.
     :param binPred: (ndarray) of shape (nFeatures,). Predicted (binary) vector.
     :param metric: (str) in ['fscore', 'prec', 'rec', 'cs', 'auc'] metric to compute.
     :param predsNegative: (bool) whether 'pred' contains negative values or not.
     :param thresholdType: (str) in ['abs', 'thres].
     :return: (float) selected metric. """

    if metric == 'fscore':
        return met.fbeta_score(binGt, binPred, beta=1)
    elif metric == 'prec':
        return met.precision_score(binGt, binPred)
    elif metric == 'rec':
        return met.recall_score(binGt, binPred)
    elif metric == 'cs':
        return cosine_similarity(gt, pred)
    elif metric == 'auc':
        if predsNegative is True:
            pred = np.abs(pred) if thresholdType == 'abs' else np.where(pred < 0, 0, pred)
        return met.roc_auc_score(binGt, pred)


def _compute_feature_importance_scores(binaryGts, binaryPreds, gts, preds, classScores, realScores, metrics,
                                       predsNegative, thresholdType, verbose):
    ret = []
    rng = np.random.default_rng(888)
    someUnifBGt, someUnifBPred, someEmptyGt, someEmptyPred = False, False, False, False
    for binGt, binPred, gt, pred in zip(binaryGts, binaryPreds, gts, preds):
        mets = []
        uniformBGt, uniformBPred, emptyGt, emptyPred = _check_correct_array_values(binGt, binPred, gt, pred)

        i = rng.integers(0, len(binGt))
        if classScores:
            if uniformBGt:
                someUnifBGt = True
                binGt[i] = int(not binGt[i])
            if uniformBPred:
                someUnifBPred = True
                binPred[i] = int(not binPred[i])
        if realScores:
            if emptyGt:
                someEmptyGt = True
                gt[i] += 1e-4
            if emptyPred:
                someEmptyPred = True
                pred[i] += 1e-4

        for metric in metrics:
            mets.append(_individual_fi_metrics(gt, pred, binGt, binPred, metric, predsNegative, thresholdType))

        ret.append(mets)

    if verbose == 1:
        if someUnifBGt:
            warnings.warn('A binary ground truth contains uniform values, so one entry has been randomly flipped '
                          'for the metrics to be defined.')
        if someUnifBPred:
            warnings.warn('A binary prediction contains uniform values, so one entry has been randomly flipped '
                          'for the metrics to be defined.')
        if someEmptyGt:
            warnings.warn('A ground truth does not contain values != 0, so 1e-4 has been added to one random entry '
                          'in both.')
        if someEmptyPred:
            warnings.warn('A prediction does not contain values != 0, so 1e-4 has been added to one random entry '
                          'in both.')

    return np.array(ret).astype(np.float32)


def feature_importance_scores(gts, preds, metrics=None, average=True, thresholdType='abs', binThreshold=0.5, verbose=1):
    """ Computes quality metrics between one or more feature importance vectors. The values in the vectors must be
    bounded in [0, 1] or [-1, 1] (to indicate negative importances in the second case). If they are not, the values will
    be mapped.

    For the computation of the precision, recall and FScore, the vectors are binarized to simulate a classification
    setting depending on the param. :code:`thresholdType`. In the case of ROC AUC, the ground truth feature importance
    vector will be binarized as in the case of 'precision', 'recall' and 'FScore' and the predicted feature importance
    vector entries will be considered as prediction scores. If the predicted vectors contain negative values, these will
    be either mapped to 0 or taken their absolute val (depending on the chosen option in the param.
    :code:`thresholdType`).

    **Edge cases**: Edge cases for when metrics are not defined have been accounted for:

        * When computing classification scores ('fscore', 'prec', 'rec'), if there is only one class in the
          ground truth and / or the prediction, one random feature will be flipped (same feature in both).
          Note that some metrics such as 'auc' may still be undefined in this case if there is only 1 feature per data
          observation.
        * For 'auc', although the ground truth is binarized, the prediction vector represents scores, and so, if both
          contain only one value, only in the ground truth a feature will be flipped. In the prediction, a small amount
          (:math:`1^{-4}`) will be summed to a random feature if no value is != 0.
        * When computing cosine similarity, if there is no value != 0 in the ground truth and / or prediction, one
          random feature will be summed 1e-4.

    **On vector ranges**: If the ground truth array or the predicted array have values that are not bounded in
    :math:`[-1, 1]` or :math:`[0, 1]`, they will be mapped accordingly. Note that if the values lie within
    :math:`[-1, 1]` or :math:`[0, 1]` no mapping will be performed, so it is assumed that the scores represent feature
    importances in those ranges. These are the cases considered for the mapping:

        * if values in the :math:`[0, \\infty]` range: map to :math:`[0, 1]`
        * if values in the :math:`[-\\infty, 0]` range: map to :math:`[-1, 1]`
        * if values in the :math:`[-\\infty, \\infty]` range: map to :math:`[-1, 1]`

    :param np.ndarray gts: (1d np.ndarray or 2d np.ndarray of shape (n_features, n_samples)) ground truth feature
        importance vectors.
    :param np.ndarray preds: (1d np.ndarray or 2d np.ndarray of shape (n_features, n_samples)) predicted feature
        importance vectors.
    :param metrics:
        (str or array-like of str) metric/s to be computed. Available metrics are

            - 'fscore': Computes the F1 Score between the ground truths and the predicted vectors.
            - 'prec': Computes the Precision Score between the ground truths and the predicted vectors.
            - 'rec': Computes the Recall Score between the ground truths and the predicted vectors.
            - 'auc': Computes the ROC AUC Score between the ground truths and the predicted vectors.
            - 'cs': Computes the Cosine Similarity between the ground truths and the predicted vectors.

        The vectors are automatically binarized for computing recall, precision and fscore.
    :param bool average: (default :code:`True`) Used only if :code:`gt` and :code:`rule` contain multiple
        observations. Should the computed metrics be averaged across all the samples?
    :param str thresholdType: Options for the binarization of the features for the computation of 'fscore', 'prec',
        'rec' and 'auc'.

            - 'abs': features with absolute val <= :code:`binThreshold` will be set to 0 and 1 otherwise. For the
              predicted feature importances in the case of 'auc', their absolute val will be taken.
            - 'thres': features <= :code:`binThreshold` will be set to 0, 1 otherwise. For the `predicted` feature
              importances in the case of 'auc', negative values will be cast to 0 and the others left *as-is*.

    :param float binThreshold:
        (in [-1, 1]) Threshold for the binarization of the features for the computation of 'fscore', 'prec', 'rec' and
        'auc'. The binarization depends on both this parameter and :code:`thresholdType`.
        If :code:`thresholdType = 'abs'`, ``binThreshold`` cannot be negative.
    :param int verbose: Verbosity level of warnings. ``1`` will report warnings, else will not.
    :return: (ndarray of shape (n_metrics,) or (n_samples, n_metrics)) specified metric/s in the indicated order. """

    if metrics is None:
        metrics = ['fscore']
    elif isinstance(metrics, str):
        metrics = [metrics]
    elif len(metrics) == 0:
        return np.array([], dtype=np.float32)

    for metric in metrics:
        if metric not in _AVAILABLE_FEATURE_IMPORTANCE_METRICS:
            raise MetricNotAvailableError(metric)

    gts, _ = scale_fi_bounds(gts)
    preds, predsNegative = scale_fi_bounds(preds)

    # binarize if necessary
    if not np.array_equal(np.unique(gts), np.array([0, 1])):
        binaryGts = _binarize_arrays(gts, method=thresholdType, threshold=binThreshold)
    else:
        binaryGts = gts.copy()
    if not np.array_equal(np.unique(preds), np.array([0, 1])):
        binaryPreds = _binarize_arrays(preds, method=thresholdType, threshold=binThreshold)
    else:
        binaryPreds = preds.copy()

    # if we have one observation, reshape it accordingly
    if len(binaryPreds.shape) == 1:
        binaryGts, binaryPreds = binaryGts.reshape(1, -1), binaryPreds.reshape(1, -1)
        gts, preds = gts.reshape(1, -1), preds.reshape(1, -1)

    # check if we are computing classification scores. This will reduce computations if ground truth vectors are
    # completely 0
    classScores, realScores = False, False
    for metric in metrics:
        if metric in ['fscore', 'prec', 'rec', 'auc']:
            classScores = True
        elif metric in ['cs']:
            realScores = True

    ret = _compute_feature_importance_scores(binaryGts, binaryPreds, gts, preds, classScores, realScores, metrics,
                                             predsNegative, thresholdType, verbose)

    if average is True and binaryPreds.shape[0] > 1:
        ret = np.mean(ret, axis=0)
    elif binaryPreds.shape[0] == 1:
        return ret.squeeze()

    return ret


def cosine_similarity(u, v, bounding: str = 'abs') -> float:
    """
    Computes cosine similarity between two real valued arrays. If negative, returns 0.

    :param u: (array-like), real valued array of dimension n.
    :param v: (array-like), real valued array of dimension n.
    :param str bounding: if the CS is < 0, bound it in [0, 1] via absolute val ('abs') or max(0, val) ('max')
    :return float: (0, 1) cosine similarity. """

    dist = 1 - cdist([u], [v], metric='cosine')[0][0]
    if bounding == 'abs':
        return np.abs(dist)
    elif bounding == 'max':
        return max(0, dist)
    else:
        raise ValueError('bounding method not valid.')
