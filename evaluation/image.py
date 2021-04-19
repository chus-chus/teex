""" Module for the evaluation of image importance (real-valued and binary) explanations. """

import cv2
import numpy as np
from matplotlib import pyplot as plt

from evaluation.featureImportance import f_score, auc_score, precision, recall
from utils.image import array_is_binary, binarize_rgb_mask, normalize_array, read_rgb_img


# todo saliency_map_score to only compute auc (rename) for saliency
# todo binary_mask_quality to be segmented into saliency_map precision, recall and fscore
# todo functions for precision, recall and fscore of binary masks (will be
#   called by the previous three after binarizing)


def saliency_map_score(gt: np.array, explanation: np.array, metric: str = 'AUC', binarizeGt: bool = True,
                       fScoreBeta: float = 1, gtBackground: str = 'white', expBackground: str = 'white',
                       **kwargs: dict) -> float:
    """
    Computes different scores of a saliency map explanation w.r.t. its ground truth explanation (a binary mask).

    :param gt: ground truth mask
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
        return binary_mask_quality(gt, explanation, metric=metric, beta=fScoreBeta, **kwargs)
    else:
        metrics = ['AUC', 'fscore', 'prec', 'rec']
        raise ValueError(f'Metric not available. Use {metrics}')


def binary_mask_quality(gt: np.array, explanation: np.array, metric: str = 'fscore', beta: float = 1, **kwargs: dict) \
        -> float:
    """
    Computes metrics for the evaluation of image binary pixel importance explanations.

    :param gt: array, ground truth pixel importance n x m binary mask.
    :param explanation: array, image pixel importance n x m binary explanation.
    :param metric: 'fscore', 'prec' or 'rec'.
    :param beta: beta value for the fscore.
    :param kwargs: extra parameters for sklearn.metrics functions.
    :return: the specified similarity metric.
    """

    if metric == 'fscore':
        return f_score(gt.flatten(), explanation.flatten(), beta=beta, average='binary', **kwargs)
    elif metric == 'prec':
        return precision(gt.flatten(), explanation.flatten(), average='binary', **kwargs)
    elif metric == 'rec':
        return recall(gt.flatten(), explanation.flatten(), average='binary', **kwargs)
    else:
        raise ValueError(f"Invalid metric. Use {['fscore', 'prec', 'rec']}")


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

    f1Score = binary_mask_quality(gtmask, testmask, metric='fscore', beta=1)
    precision = binary_mask_quality(gtmask, testmask, metric='prec')
    recall = binary_mask_quality(gtmask, testmask, metric='rec')

    print(f'F1Score: {f1Score}, Precision: {precision}, Recall: {recall}')
