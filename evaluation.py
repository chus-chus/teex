""" Module for the evaluation of explanations. """

import numpy as np
import cv2
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist
from sklearn.metrics import fbeta_score, recall_score, precision_score

from image_utils import binarize_rgb_img, read_rgb_img


def feature_importance_similarity(u: np.array, v: np.array) -> float:
    """
    Computes cosine similarity between two real valued vectors. If negative, returns 0.
     
    :param u: array, first real valued vector of dimension n.
    :param v: array, second real valued vector of dimension n.
    :return: (0, 1) cosine similarity.
    """
    return max(0, cdist(u, v, metric='cosine')[0][0])


def pixel_importance_similarity(gt: np.array, image: np.array, metric: str = 'fscore', beta: float = 1,
                                **kwargs: dict) -> float:
    """
    Metrics for the evaluation of image pixel importance explanations.

    :param gt: array, ground truth pixel importance n x m binary mask.
    :param image: array, image pixel importance n x m binary explanation.
    :param metric: 'fscore', 'prec' or 'rec'.
    :param beta: beta value for the fscore.
    :param kwargs: extra parameters for sklearn.metrics functions.
    :return: the specified similarity metric.
    """

    if metric == 'fscore':
        return fbeta_score(gt.flatten(), image.flatten(), beta=beta, average='binary', **kwargs)
    elif metric == 'prec':
        return precision_score(gt.flatten(), image.flatten(), average='binary', **kwargs)
    elif metric == 'rec':
        return recall_score(gt.flatten(), image.flatten(), average='binary', **kwargs)
    else:
        metrics = ['fscore', 'prec', 'rec']
        raise ValueError(f'Invalid metric. Available: {metrics}')


if __name__ == '__main__':
    # testing
    imgpath = '../TAIAOexp_CLI/data/Kahikatea/data/ve_positive/Biospatial_20160702U_1860_01_03.png'

    img = read_rgb_img(imgpath)

    maskpath = '../TAIAOexp_CLI/data/Kahikatea/expl/ve_positive/Biospatial_20160702U_1860_01_03.png'

    # load ground truth mask and generate perturbed mask for testing
    gtmask = read_rgb_img(maskpath)

    plt.imshow(gtmask)
    plt.show()

    gtmask = binarize_rgb_img(gtmask, normalize=True)

    plt.imshow(gtmask)
    plt.show()

    # perturb ground truth mask
    testmask = cv2.erode(gtmask, np.ones((5, 5), np.uint8))

    plt.imshow(testmask)
    plt.show()

    f1Score = pixel_importance_similarity(gtmask, testmask, metric='fscore', beta=1)
    precision = pixel_importance_similarity(gtmask, testmask, metric='prec')
    recall = pixel_importance_similarity(gtmask, testmask, metric='rec')

    print(f'F1Score: {f1Score}, Precision: {precision}, Recall: {recall}')

