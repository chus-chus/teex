""" Module for evaluation of word importance explanations. """

from typing import List, Union, Dict

import numpy as np

from teex._utils._errors import MetricNotAvailableError
from teex.featureImportance.eval import feature_importance_scores, _AVAILABLE_FEATURE_IMPORTANCE_METRICS

_AVAILABLE_WORD_IMPORTANCE_METRICS = {'prec', 'rec', 'fscore', 'cs', 'auc'}


def _get_reference_words(gts, preds):
    """ Get reference words as the union of the words in gts and preds. """

    vocabWords = []
    if isinstance(gts, dict):
        vocabWords = list({**gts, **preds}.keys())
    elif isinstance(gts, (list, tuple, np.ndarray)):
        vocabWords = [list({**gt, **pred}.keys()) for gt, pred in zip(gts, preds)]

    return vocabWords


def word_importance_scores(gts: Union[Dict[str, float], List[Dict[str, float]]],
                           preds: Union[Dict[str, float], List[Dict[str, float]]],
                           vocabWords: Union[List[str], List[List[str]]] = None,
                           metrics: Union[str, List[str]] = None,
                           binThreshold: float = .5,
                           average: bool = True) -> np.ndarray:
    """ Quality metrics for word importance explanations, where each word is considered as a feature. An example of
    an explanation:

    >>> {'skate': 0.7, 'to': 0.2, 'me': 0.5}

    :param gts: (dict, array-like of dicts) ground truth word importance/s, where each BOW is represented as a
        dictionary with words as keys and floats as importances. Importances must be in :math:`[0, 1]` or +
        :math:`[-1, 1]`.
    :param preds: (dict, array-like of dicts) predicted word importance/s, where each BOW is represented as a
        dictionary with words as keys and floats as importances. Importances must be in the same scale as param.
        ``gts``.
    :param vocabWords: (array-like of str 1D or 2D for multiple reference vocabularies, default None) Vocabulary words.
        If ``None``, the union of the words in each ground truth and predicted explanation will be interpreted as the
        vocabulary words. This is needed for when explanations are converted to feature importance vectors. If this
        parameter is provided as a 1D list, the vocabulary words will be the same for all explanations, but if not
        provided or given as a 2D array-like (same number of reference vocabularies as there are explanations),
        different vocabulary words will be considered for each explanation.
    :param metrics: (str / array-like of str, default=['prec']) Quality metric/s to compute. Available:

        - All metrics in :func:`teex.featureImportance.eval.feature_importance_scores`.

    :param float binThreshold: (in [0, 1], default .5) pixels of images in :code:`sMaps` with a val bigger than this
        will be set to 1 and 0 otherwise when binarizing for the computation of 'fscore', 'prec', 'rec' and 'auc'.
    :param bool average: (default :code:`True`) Used only if :code:`gts` and :code:`preds` contain multiple
        observations. Should the computed metrics be averaged across all samples?
    :return: specified metric/s in the original order. Can be of shape:

        - (n_metrics,) if only one image has been provided in both :code:`gts` and :code:`preds` or when both are
          contain multiple observations and :code:`average=True`.
        - (n_metrics, n_samples) if :code:`gts` and :code:`preds` contain multiple observations and
          :code:`average=False`.

    :rtype: np.ndarray """

    if metrics is None:
        metrics = ['prec']
    elif isinstance(metrics, str):
        metrics = [metrics]

    fiGts, fiPreds = None, None
    metricTypes = []  # fi, r (regular)
    for metric in metrics:
        if metric not in _AVAILABLE_WORD_IMPORTANCE_METRICS:
            raise MetricNotAvailableError(metric)
        elif metric in _AVAILABLE_FEATURE_IMPORTANCE_METRICS:
            if vocabWords is None:
                vocabWords = _get_reference_words(gts, preds)
            fiGts = word_to_feature_importance(gts, vocabWords)
            fiPreds = word_to_feature_importance(preds, vocabWords)
            metricTypes.append('fi')
        else:
            metricTypes.append('r')

    fiMetrics = [metrics[i] for i, metricType in enumerate(metricTypes) if metricType == 'fi']
    # regMetrics = [metrics[i] for i, metricType in enumerate(metricTypes) if metricType == 'r']

    res = []
    if len(fiMetrics) != 0:
        # multiple vocabularies (gts or preds possibly of different lengths)
        if not isinstance(vocabWords[0], str):
            i = 0
            for gt, pred in zip(fiGts, fiPreds):
                verbose = 1 if i == 0 else 0
                res.append(feature_importance_scores(gt, pred, metrics=fiMetrics, average=False,
                           binThreshold=binThreshold, verbose=verbose))
                i = 1
            res = np.array(res)
            if average is True:
                res = np.mean(res, axis=0)
        else:
            # noinspection PyTypeChecker
            res = feature_importance_scores(fiGts, fiPreds, metrics=fiMetrics, average=average,
                                            binThreshold=binThreshold)

    # if len(regMetrics) != 0:
    #     if isinstance(gts, dict):
    #         if not isinstance(preds, dict):
    #             raise IncompatibleGTAndPredError
    #         # only 1 observation, return feature importance score or and custom score.
    #         raise NotImplementedError('Custom scores not implemented yet.')
    #     elif isinstance(gts, (list, np.ndarray, tuple)):
    #         if not isinstance(preds, (list, np.ndarray, tuple)):
    #             raise IncompatibleGTAndPredError
    #         # multiple obs, return feature importance scores or and custom scores.
    #         raise NotImplementedError('Custom scores not implemented yet.')
    #     else:
    #         raise TypeError("Ground truth type not supported.")

    return res


def word_to_feature_importance(wordImportances, vocabWords) -> list:
    """ Maps words with importance weights into a feature importance vector.

    :param wordImportances: (dict or array-like of dicts) words with feature importances as values with the same format
        as described in the method :func:`word_importance_scores`.
    :param vocabWords: (array-like of str, 1D or 2D for multiple reference vocabularies) :math:`m` words that
        should be taken into account when transforming into vector representations. Their order will be followed.
    :return: Word importances as feature importance vectors. Return types:

        - list of np.ndarray, if multiple vocabularies because of the possible difference in size of the reference
          vocabularies in each explanation.
        - np.ndarray, if only 1 vocabulary

    :Example:

    >>> word_to_feature_importance({'a': 1, 'b': .5},['a', 'b', 'c'])
    >>> [1, .5, 0]
    >>> word_to_feature_importance([{'a': 1, 'b': .5}, {'b': .5, 'c': .9}],['a', 'b', 'c'])
    >>> [[1, .5, 0. ], [0, .5, .9]]
    """

    if isinstance(wordImportances, (list, np.ndarray, tuple)):
        res = []
        # check if we have multiple reference vocabularies
        if isinstance(vocabWords[0], (list, tuple, np.ndarray)):
            for i, wordDict in enumerate(wordImportances):
                res.append(np.array([wordDict[word] if word in wordDict else 0 for word in vocabWords[i]],
                                    dtype=np.float32))
        else:
            for wordDict in wordImportances:
                res.append(np.array([wordDict[word] if word in wordDict else 0 for word in vocabWords],
                                    dtype=np.float32))
    elif isinstance(wordImportances, dict):
        res = np.array([wordImportances[word] if word in wordImportances else 0 for word in vocabWords],
                       dtype=np.float32)
    else:
        raise ValueError('The BOW is not a dict or array-like of dicts.')
    return res
