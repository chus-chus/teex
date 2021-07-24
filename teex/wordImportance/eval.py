""" Module for evaluation of word importance explanations. """

from typing import List, Union, Dict

import numpy as np

from teex._utils._errors import MetricNotAvailableError, IncompatibleGTAndPredError
from teex.featureImportance.eval import feature_importance_scores

_AVAILABLE_WORD_IMPORTANCE_METRICS = {'prec', 'rec', 'fscore'}


def word_importance_scores(gts: Union[Dict[str, float], List[Dict[str, float]]],
                           preds: Union[Dict[str, float], List[Dict[str, float]]],
                           metrics: Union[str, List[str]] = None,
                           average: bool = True) -> Union[float, np.ndarray]:
    """ Quality metrics for word importance explanations, where each word is considered as a feature.

        :param gts: (dict, array-like of dicts) ground truth word importance/s, where each BOW is represented as a
            dictionary with words as keys and floats as importances. Importances must be in :math:`[0, 1]` or +
            :math:`[-1, 1]`.
        :param preds: (dict, array-like of dicts) predicted word importance/s, where each BOW is represented as a
            dictionary with words as keys and floats as importances. Importances must be in the same scale as param.
            ``gts``.
        :param metrics: (str / array-like of str, default=['auc']) Quality metric/s to compute. Available:
            # todo metrics documentation after they are implemented

        :param float binThreshold: (in [0, 1]) pixels of images in :code:`sMaps` with a val bigger than this will be set
            to 1 and 0 otherwise when binarizing for the computation of 'fscore', 'prec', 'rec' and 'auc'.
        :param bool average: (default :code:`True`) Used only if :code:`gts` and :code:`preds` contain multiple
            observations. Should the computed metrics be averaged across all samples?
        :return: specified metric/s in the original order. Can be of shape
            - (n_metrics,) if only one image has been provided in both :code:`gts` and :code:`preds` or when both are
              contain multiple observations and :code:`average=True`.
            - (n_metrics, n_samples) if :code:`gts` and :code:`preds` contain multiple observations and
              :code:`average=False`.
        :rtype: np.ndarray

        """

    if metrics is None:
        metrics = ['fscore']
    elif isinstance(metrics, str):
        metrics = [metrics]

    for metric in metrics:
        if metric not in _AVAILABLE_WORD_IMPORTANCE_METRICS:
            raise MetricNotAvailableError(metric)

    if isinstance(gts, dict):
        if not isinstance(preds, dict):
            raise IncompatibleGTAndPredError
        # only 1 observation, return feature importance score or and custom score.
        return 1
    elif isinstance(gts, (list, np.ndarray, tuple)):
        if not isinstance(preds, (list, np.ndarray, tuple)):
            raise IncompatibleGTAndPredError
        # multiple obs, return feature importance scores or and custom scores.
    else:
        raise TypeError("Ground truth type not supported.")
