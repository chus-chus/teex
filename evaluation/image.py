""" Module for the evaluation of image importance (real-valued and binary) explanations. """

import cv2
import numpy as np
from matplotlib import pyplot as plt

from evaluation.featureImportance import f_score, auc_score, precision, recall
from utils.image import array_is_binary, binarize_rgb_mask, normalize_array, read_rgb_img


def saliency_map_score(gt: np.array, explanation: np.array, metric: str = 'AUC', binarizeGt: bool = True,
                       fScoreBeta: float = 1, gtBackground: str = 'white', expBackground: str = 'white',
                       **kwargs: dict) -> float:
    """
    Computes different scores of a saliency map explanation w.r.t. its ground truth explanation (a binary mask).
    The saliency map should not be binary. If it is, use the 'binary_mask_scores' method instead.

    :param gt: ground truth binary mask
    :param explanation: saliency map explanation (non-salient areas must be 0-valued.)
    :param metric: Quality metric to compute: {'AUC', 'fscore', 'prec', 'rec'}
    :param binarizeGt: Should the g.t. mask provided be binarized?
    :param fScoreBeta: Beta for the fscore in case of it being computed.
    :param gtBackground: Color of the background of the g.t. mask ('black' for 0, 'white' for 255)
    :param expBackground: Color of the background of the explanation ('black' for 0, 'white' for 255)
    :param kwargs: Extra arguments for 'pixel_importance_mask_quality'
    :return: desired metric
    """
    if not array_is_binary(gt) and not binarizeGt:
        raise ValueError('Ground truth is not binary: binarize before calling.')
    elif binarizeGt:
        gt = binarize_rgb_mask(gt, bgColor=gtBackground)

    if metric == 'AUC':
        # saliency map's background has to be 0
        explanation = normalize_array(explanation)
        return auc_score(gt.flatten(), explanation.flatten())
    elif metric in ['fscore', 'prec', 'rec']:
        explanation = binarize_rgb_mask(explanation, bgColor=expBackground)
        return binary_mask_scores(gt, explanation, metric=metric, beta=fScoreBeta, **kwargs)
    else:
        metrics = ['AUC', 'fscore', 'prec', 'rec']
        raise ValueError(f'Metric not available. Use {metrics}')


def binary_mask_scores(gt: np.array, explanation: np.array, metrics=None, beta: float = 1, **kwargs: dict) \
        -> float:
    """
    Computes metrics for the evaluation of image binary pixel importance explanations.

    :param gt: array, ground truth pixel importance n x m binary mask.
    :param explanation: array, image pixel importance n x m binary explanation.
    :param metrics: 'fscore', 'prec' or 'rec'.
    :param beta: beta value for the fscore.
    :param kwargs: extra parameters for sklearn.metrics functions.
    :return: (float / list) the specified similarity metric/s.
    """

    if metrics is None:
        metrics = ['fscore']

    gt = gt.flatten()
    explanation = explanation.flatten()

    ret = []
    for metric in metrics:
        if metric == 'fscore':
            ret.append(f_score(gt, explanation, beta=beta, average='binary', **kwargs))
        elif metric == 'prec':
            ret.append(precision(gt, explanation, average='binary', **kwargs))
        elif metric == 'rec':
            ret.append(recall(gt, explanation, average='binary', **kwargs))
        else:
            raise ValueError(f"Invalid metric. Use {['fscore', 'prec', 'rec']}")

    if len(ret) == 1:
        return ret[0]
    return ret

# todo implement in featureImportance evaluation a method for computing the metrics all at one (same as here or in
#  evaluation/rule) and make those two call the featureImportance method. Saliency map scores calls binary mask scores
#  if one ones to compute any metric other than AUC and also returns an array of metrics.


if __name__ == '__main__':
    imgpath = '../TAIAOexp_CLI/data/Kahikatea/data/ve_positive/Biospatial_20160702U_1860_01_03.png'

    img = read_rgb_img(imgpath)

    maskpath = '../TAIAOexp_CLI/data/Kahikatea/expl/ve_positive/Biospatial_20160702U_1860_01_03.png'

    # load ground truth mask and generate perturbed mask for testing
    gtmask = read_rgb_img(maskpath)

    plt.imshow(gtmask)
    plt.show()

    gtmask = binarize_rgb_mask(gtmask, bgColor='white')

    plt.imshow(gtmask)
    plt.show()

    # perturb ground truth mask
    testmask = cv2.erode(gtmask, np.ones((5, 5), np.uint8))

    plt.imshow(testmask)
    plt.show()

    f1Score = binary_mask_scorse(gtmask, testmask, metric='fscore', beta=1)
    precision = binary_mask_scores(gtmask, testmask, metric='prec')
    recall = binary_mask_scores(gtmask, testmask, metric='rec')

    print(f'F1Score: {f1Score}, Precision: {precision}, Recall: {recall}')
