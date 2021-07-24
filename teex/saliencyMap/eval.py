""" Module for evaluation of saliency map explanations. """

import warnings
import numpy as np

from teex._utils._errors import MetricNotAvailableError
from teex.featureImportance.eval import feature_importance_scores
from teex.saliencyMap.data import binarize_rgb_mask

_AVAILABLE_SALIENCY_MAP_METRICS = {'fscore', 'prec', 'rec', 'cs', 'auc'}


def saliency_map_scores(gts, sMaps, metrics=None, binThreshold=0.01, gtBackgroundVals='high', average=True):
    """ Quality metrics for saliency map explanations, where each pixel is considered as a feature.
    Computes different scores of a saliency map explanation w.r.t. its ground truth explanation (a binary mask).

    :param np.ndarray gts: ground truth RGB or binary mask/s. Accepted shapes are

            - *(imageH, imageW)* A single grayscale mask, where each pixel should be 1 if it is part of the salient
              class and 0 otherwise.
            - *(imageH, imageW, 3)* A single RGB mask, where pixels that **do not** contain the salient class are all
              either black (all channels set to 0) or white (all channels set to max.).
            - *(nSamples, imageH, imageW)* Multiple grayscale masks, where for each where, in each image, each pixel
              should be 1 if it is part of the salient class and 0 otherwise.
            - *(nSamples, imageH, imageW, 3)* Multiple RGB masks, where for each image, pixels that *do not* contain
              the salient class are all either black (all channels set to 0) or white (all channels set to max.).

        If the g.t. masks are RGB they will be binarized (see param :code:`gtBackground` to specify the color of the
        pixels that pertain to the non-salient class).
    :param np.ndarray sMaps: grayscale saliency map explanation/s ([0, 1] or [-1, 1] normalised). Supported shapes are

        - *(imageH, imageW)* A single explanation
        - *(nSamples, imageH, imageW)* Multiple explanations

    :param metrics: (str / array-like of str, default=['auc']) Quality metric/s to compute. Available:

        - 'auc': ROC AUC score. The val of each pixel of each saliency map in :code:`sMaps` is considered as a
          prediction probability of the pixel pertaining to the salient class.
        - 'fscore': F1 Score.
        - 'prec': Precision Score.
        - 'rec': Recall score.
        - 'cs': Cosine Similarity.

        For 'fscore', 'prec', 'rec' and 'cs', the saliency maps in :code:`sMaps` are binarized (see param
        :code:`binThreshold`).
    :param float binThreshold:
        (in [0, 1]) pixels of images in :code:`sMaps` with a val bigger than this will be set to 1 and 0 otherwise
        when binarizing for the computation of 'fscore', 'prec', 'rec' and 'auc'.
    :param str gtBackgroundVals: Only used when provided ground truth explanations are RGB. Color of the background
        of the g.t. masks 'low' if pixels in the mask representing the non-salient class are dark, 'high' otherwise).
    :param bool average: (default :code:`True`) Used only if :code:`gts` and :code:`sMaps` contain multiple
        observations. Should the computed metrics be averaged across all of the samples?
    :return: specified metric/s in the original order. Can be of shape

        - *(n_metrics,)* if only one image has been provided in both :code:`gts` and :code:`sMaps` or when both are
          contain multiple observations and :code:`average=True`.
        - *(n_metrics, n_samples)* if :code:`gts` and :code:`sMaps` contain multiple observations and
          :code:`average=False`.
    :rtype: np.ndarray
    """

    if metrics is None:
        metrics = ['auc']
    elif isinstance(metrics, str):
        metrics = [metrics]

    for metric in metrics:
        if metric not in _AVAILABLE_SALIENCY_MAP_METRICS:
            raise MetricNotAvailableError(metric)

    if len(gts.shape) == 2:
        # Single binary g.t. mask
        return feature_importance_scores(gts.flatten(), sMaps.flatten(), metrics=metrics, binThreshold=binThreshold)
    elif len(gts.shape) == 3:
        # naive RGB check
        if gts.shape[2] == 3:
            warnings.warn('Binarizing g.t. RGB mask. If it is not RGB, binarize it before calling this function.')
            gts = binarize_rgb_mask(gts, bgValue=gtBackgroundVals)
            return feature_importance_scores(gts.flatten(), sMaps.flatten(), metrics=metrics, binThreshold=binThreshold)
        else:
            # multiple binary masks provided
            return feature_importance_scores(gts.reshape(gts.shape[0], gts.shape[1] * gts.shape[2]),
                                             sMaps.reshape(sMaps.shape[0], sMaps.shape[1] * sMaps.shape[2]),
                                             metrics=metrics, average=average, binThreshold=binThreshold)
    # multiple RGB masks provided
    elif len(gts.shape) == 4:
        warnings.warn(f'Binarizing {gts.shape[0]} g.t. RGB masks.')
        newGts = np.zeros((gts.shape[0], gts.shape[1], gts.shape[2]))
        for imIndex in range(gts.shape[0]):
            newGts[imIndex] = binarize_rgb_mask(gts[imIndex], bgValue=gtBackgroundVals)
        return feature_importance_scores(newGts.reshape(newGts.shape[0], newGts.shape[1]*newGts.shape[2]),
                                         sMaps.reshape(sMaps.shape[0], sMaps.shape[1]*sMaps.shape[2]),
                                         metrics=metrics, average=average, binThreshold=binThreshold)
    else:
        raise ValueError(f'Shape {gts.shape} of ground truth explanations not supported.')
