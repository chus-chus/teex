""" Module for the evaluation of image importance (real-valued and binary) explanations. """

from evaluation.featureImportance import feature_importance_scores
from utils.image import array_is_binary, binarize_rgb_mask, normalize_array, read_rgb_img, is_rgb


def saliency_map_scores(gt, explanation, metrics='auc', binarizeGt=True, gtBackground='light', **kwargs) -> float:
    """
    Computes different scores of a saliency map explanation w.r.t. its ground truth explanation (a binary mask).
    The saliency map should not be binary. If it is, use the 'binary_mask_scores' method instead.

    :param gt: (ndarray) ground truth RGB or grayscale mask
    :param explanation: (ndarray) grayscale (1 channel) saliency map explanation (non-salient areas must be 0-valued.)
    :param metrics: (str / array-like) Quality metric/s to compute: {'auc', 'fscore', 'prec', 'rec', 'cs'}
    :param binarizeGt: (bool) Should the g.t. mask provided be binarized? (should be True if mask is RGB)
    :param gtBackground: (str) Color of the background of the g.t. mask ('dark' for low values, 'light' for high values)
                               Ignored if gt is grayscale.
    :return: (list) metric/s
    """
    if not array_is_binary(gt) and not binarizeGt:
        raise ValueError('Ground truth is not binary: binarize before calling.')
    elif binarizeGt:
        if is_rgb(gt):
            gt = binarize_rgb_mask(gt, bgColor=gtBackground)
        else:
            gt[gt != 0] = 1

    explanation = normalize_array(explanation)
    return feature_importance_scores(gt.flatten(), explanation.flatten(), metrics=metrics, binarizeExp=True, **kwargs)


def binary_mask_scores(gt, explanation, metrics=None, **kwargs) -> float:
    """
    Computes metrics for the evaluation of image binary pixel importance explanations.

    :param gt: (ndarray), ground truth pixel importance n x m binary mask.
    :param explanation: (ndarray), image pixel importance n x m binary explanation.
    :param metrics: (str / array-like) 'fscore', 'prec', 'rec', 'auc', 'cs'
    :return: (list) the specified similarity metric/s.
    """

    return feature_importance_scores(gt.flatten(), explanation.flatten(), metrics=metrics, binarizeExp=False, **kwargs)


if __name__ == '__main__':
    import numpy as np

    from cv2 import erode
    from matplotlib import pyplot as plt

    imgpath = '../TAIAOexp_CLI/data/Kahikatea/data/ve_positive/Biospatial_20160702U_1860_01_03.png'

    img = read_rgb_img(imgpath)

    maskpath = '../TAIAOexp_CLI/data/Kahikatea/expl/ve_positive/Biospatial_20160702U_1860_01_03.png'

    # load ground truth mask and generate perturbed mask for testing
    gtmask = read_rgb_img(maskpath)

    plt.imshow(gtmask)
    plt.show()

    gtmask = binarize_rgb_mask(gtmask, bgColor='light')

    plt.imshow(gtmask)
    plt.show()

    # perturb ground truth mask
    testmask = erode(gtmask, np.ones((5, 5), np.uint8))

    plt.imshow(testmask)
    plt.show()

    f1Score, prec, rec, auc, cs = binary_mask_scores(gtmask, testmask, metrics=['fscore', 'prec', 'rec', 'auc', 'cs'])

    print(f'F1Score: {f1Score}, Precision: {prec}, Recall: {rec}, AUC: {auc}, CS: {cs}')
