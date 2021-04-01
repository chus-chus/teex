""" Module for the evaluation of explanations. """

import numpy as np
import cv2
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist
from sklearn.metrics import fbeta_score, recall_score, precision_score, roc_auc_score

from image_utils import read_rgb_img, binarize_rgb_mask, normalize_array, array_is_binary


def feature_importance_similarity(u: np.array, v: np.array) -> float:
    """
    Computes cosine similarity between two real valued vectors. If negative, returns 0.
     
    :param u: array, first real valued vector of dimension n.
    :param v: array, second real valued vector of dimension n.
    :return: (0, 1) cosine similarity.
    """
    return max(0, cdist(u, v, metric='cosine')[0][0])


def pixel_importance_mask_quality(gt: np.array, explanation: np.array, metric: str = 'fscore', beta: float = 1,
                                  **kwargs: dict) -> float:
    """
    Metrics for the evaluation of image pixel importance explanations.

    :param gt: array, ground truth pixel importance n x m binary mask.
    :param explanation: array, image pixel importance n x m binary explanation.
    :param metric: 'fscore', 'prec' or 'rec'.
    :param beta: beta value for the fscore.
    :param kwargs: extra parameters for sklearn.metrics functions.
    :return: the specified similarity metric.
    """

    if metric == 'fscore':
        return fbeta_score(gt.flatten(), explanation.flatten(), beta=beta, average='binary', **kwargs)
    elif metric == 'prec':
        return precision_score(gt.flatten(), explanation.flatten(), average='binary', **kwargs)
    elif metric == 'rec':
        return recall_score(gt.flatten(), explanation.flatten(), average='binary', **kwargs)
    else:
        metrics = ['fscore', 'prec', 'rec']
        raise ValueError(f'Invalid metric. Use {metrics}')


def saliency_map_quality(gt: np.array, explanation: np.array, metric: str = 'AUC', binarizeGt: bool = True,
                         fScoreBeta: float = 1, gtBackground: str = 'white', expBackground: str = 'white',
                         **kwargs: dict) -> float:
    """
    Computes the quality of a saliency map explanation w.r.t. its ground truth explanation (a binary mask).
    :param gt: ground truth mask
    :param explanation: saliency map explanation
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
        return roc_auc_score(gt.flatten(), explanation.flatten())
    elif metric in ['fscore', 'prec', 'rec']:
        explanation = binarize_rgb_mask(explanation, bgColor=expBackground)
        return pixel_importance_mask_quality(gt, explanation, metric=metric, beta=fScoreBeta, **kwargs)
    else:
        metrics = ['AUC', 'fscore', 'prec', 'rec']
        raise ValueError(f'Metric not available. Use {metrics}')


if __name__ == '__main__':
    # testing
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

    f1Score = pixel_importance_mask_quality(gtmask, testmask, metric='fscore', beta=1)
    precision = pixel_importance_mask_quality(gtmask, testmask, metric='prec')
    recall = pixel_importance_mask_quality(gtmask, testmask, metric='rec')

    print(f'F1Score: {f1Score}, Precision: {precision}, Recall: {recall}')

