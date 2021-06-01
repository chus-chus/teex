""" The image module. Contains all of the methods regarding this explanation type: data generation, evaluation of
explanations and other utils. """

import numpy as np
import random

from evaluation.featureImportance import feature_importance_scores
from transparentModels.baseClassifier import BaseClassifier
from _utils import _array_is_binary, _is_rgb, _binarize_rgb_mask, _normalize_array, _read_rgb_img

_AVAILABLE_SALIENCY_MAP_METRICS = {'fscore', 'prec', 'rec', 'cs', 'auc'}

# ===================================
#       TRANSPARENT MODEL
# ===================================


class TransparentImageClassifier(BaseClassifier):
    """ Transparent, pixel-based classifier with pixel (features) importances as explanations. Predicts the
    class of the images based on whether they contain a certain specified pattern or not. Class 1 if they contain
    the pattern, 0 otherwise. To be trained only a pattern needs to be fed. Follows the sklean API. Presented in
    [Evaluating local explanation methods on ground truth, Riccardo Guidotti, 2021].

    IMPORTANT NOTE ON GENERATING G.T. EXPLANATIONS: the ground truth explanations are automatically generated with
    the 'dataGen.gen_image_data' method, so using this class to generate g.t. explanations is not recommended,
    as it is much less efficient. """

    def __init__(self):
        super().__init__()
        self.pattern = None
        self.pattH = None
        self.pattW = None
        self.cellH = None
        self.cellW = None
        self._binaryPattern = None
        self._patternMask = None
        self._maskedPattern = None

    def fit(self, pattern, cellH=1, cellW=1):
        self.pattern = pattern.astype(np.float32)
        self.pattH, self.pattW = pattern.shape[0], pattern.shape[1]
        self.cellH, self.cellW = cellH, cellW
        self._binaryPattern = np.squeeze(np.where(np.delete(pattern, (0, 1), 2) != 0, 1, 0))
        # noinspection PyArgumentList
        self._patternMask = self._binaryPattern.astype(bool).reshape(self.pattH, self.pattW, 1)
        self._patternMask = np.concatenate((self._patternMask, self._patternMask, self._patternMask), axis=2)
        self._maskedPattern = self.pattern[self._patternMask]

    def predict(self, obs):
        """ Predicts the class for each observation.
            :param obs: array of n images as ndarrays of np.float32 type.
            :return: array of n predicted labels. """
        ret = []
        for image in obs:
            ret.append(1) if self._has_pattern(image) else ret.append(0)
        return ret

    def predict_proba(self, obs):
        """ Predicts probability that each observation belongs to class 1 or 0. Probability of class 1 will be 1 if
            the image contains the pattern and 0 otherwise.
            :param obs: array of n images as ndarrays.
            :return: array of n probability tuples of length 2 ª"""
        ret = []
        for image in obs:
            if self._has_pattern(image):
                ret.append([0., 1.])
            else:
                ret.append([1., 0.])
        return ret

    def explain(self, obs):
        """ Explain observations' predictions with binary masks (pixel importance arrays).
            :param obs: array of n images as ndarrays.
            :return: list with n binary masks as explanations. """
        exps = []
        for image in obs:
            hasPat, indices = self._has_pattern(image, retIndices=True)
            exp = np.zeros((image.shape[0], image.shape[1]))
            if hasPat:
                exp[indices[0]:(indices[0]+self.pattH), indices[1]:(indices[1]+self.pattW)] = self._binaryPattern
            exps.append(exp)
        return exps

    def _has_pattern(self, image, retIndices=False):
        """ Searches for the pattern in the image and returns whether it contains it or not and its upper left indices
        if specified. The pattern is contained within an image if the distribution and color of the cells != 0 coincide.
        """

        hasPat = False
        indices = (0, 0)
        for row in range(0, len(image[0]) - self.pattH, self.cellH):
            for col in range(0, len(image[1]) - self.pattW, self.cellW):
                if (image[row:(row+self.pattH), col:(col+self.pattW)][self._patternMask] == self._maskedPattern).all():
                    hasPat = True
                    indices = (row, col)
                    break
            if hasPat:
                break
        if retIndices:
            return hasPat, indices
        else:
            return hasPat


# ===================================
#       DATA GENERATION
# ===================================

def gen_image_data(method='seneca', nSamples=1000, imageH=32, imageW=32, patternH=16, patternW=16, cellH=4,
                   cellW=4, patternProp=0.5, fillPct=0.4, colorDev=0.1, randomState=888, returnModel=False):
    """ Generate synthetic classification image data with ground truth explanations.
    # todo add kahikatea

     :param method: (str) method to use for the generation of the data and g.t. explanations. Available:
        - **'seneca'**: The g.t. explanations generated are binary ndarray masks of shape (imageH, imageW) that indicate
        the position of the pattern in an image (zero array if the pattern is not present) and are
        generated following the procedure presented in [Evaluating local explanation methods on ground truth, Riccardo
        Guidotti, 2021]. The generated RGB images belong to one class if they contain a certain generated pattern and to
        the other if not. The images are composed of homogeneous cells of size (cellH, cellW), which in turn compose a
        certain pattern of shape (patternH, patternW) that is inserted on some of the generated images.

     :param nSamples: (int) number of images to generate.
     :param imageH: (int) height in pixels of the images. For :code:`method='seneca'` must be multiple of cellH.
     :param imageW: (int) width in pixels of the images. For :code:`method='seneca'` must be multiple of cellW.
     :param patternH: (int) height in pixels of the pattern. Must be <= imageH and multiple of cellH.
     Only used for :code:`method='seneca'`.
     :param patternW: (int) width in pixels of the pattern. Must be <= imageW and multiple of cellW.
     Only used for :code:`method='seneca'`.
     :param cellH: (int) height in pixels of each cell. Only used for :code:`method='seneca'`.
     :param cellW: (int) width in pixels of each cell. Only used for :code:`method='seneca'`.
     :param patternProp: (float, [0, 1]) percentage of appearance of the pattern in the dataset. Only used for
     :code:`method='seneca'`.
     :param fillPct: (float, [0, 1]) percentage of cells filled (not black) in each image. Only used for
     :code:`method='seneca'`.
     :param colorDev: (float, [0, 0.5]) maximum value summed to 0 valued channels and minimum value substracted to 1
                      valued channels of filled cells. If 0, each cell will be completely red, green or blue.
                      If > 0, colors may be a mix of the three channels (one ~1, the other two ~0). Only used for
                      :code:`method='seneca'`.
     :param randomState: (int) random seed.
     :param returnModel: (bool) should a transparent image classifier trained with the data be returned? Only used for
     :code:`method='seneca'`.
     :return:
        - X (ndarray) of shape (nSamples, imageH, imageW, 3). Generated image data.
        - y (ndarray) of shape (nSamples,). Image labels. 1 if an image contains the pattern and 0 otherwise.
        - explanations (ndarray) of shape (nSamples, imageH, imageW). Generated ground truth explanations.
        - pattern (ndarray) of shape (patternH, patternW, 3). Only returned if :code:`method='seneca'`.
        - model (:class:`rule.TransparentImageClassifier`) Model instance trained to recognize the pattern in the data
        (returned if :code:`returnModel` is True only when :code:`method='seneca'`). """

    methods = ['seneca']

    if method not in methods:
        raise ValueError(f'Method not available. Use one in {methods}')

    random.seed(randomState)
    rng = np.random.default_rng(randomState)

    if method == 'seneca':
        if imageH % patternH != 0 or imageW % patternW != 0 or imageH % cellH != 0 or imageW % cellW != 0:
            raise ValueError('Image height and widht not multiple of cell or pattern dimensions.')
        if imageH < patternH or imageH < cellH or imageW < patternW or imageW < cellW or patternH < cellH or \
                patternW < cellW:
            raise ValueError('Cells should be smaller than patterns and patterns than image size.')

        nWithPattern = round(nSamples * patternProp)
        pattern = _generate_image_seneca(imageH=patternH, imageW=patternW, cellH=cellH, cellW=cellW, fillPct=fillPct,
                                         rng=rng, colorDev=colorDev, pattern=None)

        # transform pattern into a 2d array and then set the channel to 1 if the pixel had any intensity at all (if a
        # cell was part of the pattern it will have at least some intensity). Squeeze it so it has shape patH x patW.
        binaryPattern = np.squeeze(np.where(np.delete(pattern, (0, 1), 2) != 0, 1, 0))

        data = []
        for _ in range(nWithPattern):
            image, explanation = _generate_image_seneca(imageH=imageH, imageW=imageW, cellH=cellH, cellW=cellW,
                                                        fillPct=fillPct, rng=rng, colorDev=colorDev, pattern=pattern,
                                                        binaryPattern=binaryPattern)
            data.append((image, explanation, 1))
        for _ in range(nSamples - nWithPattern):
            image = _generate_image_seneca(imageH=imageH, imageW=imageW, cellH=cellH, cellW=cellW, fillPct=fillPct,
                                           rng=rng, colorDev=colorDev, pattern=None)
            # blank explanation
            explanation = np.zeros((imageH, imageW))
            data.append((image, explanation, 0))

        random.shuffle(data)
        imgs, exps, labels = zip(*data)

        if returnModel:
            mod = TransparentImageClassifier()
            mod.fit(pattern, cellH=cellH, cellW=cellW)
            return np.array(imgs, dtype=np.float32), np.array(labels, dtype=int), np.array(exps, dtype=int), pattern, \
                mod
        else:
            return np.array(imgs, dtype=np.float32), np.array(labels, dtype=int), np.array(exps, dtype=int), pattern


def _generate_image_seneca(imageH, imageW, cellH, cellW, fillPct, rng, colorDev, pattern=None, binaryPattern=None):
    """ Generates RGB image as ndarray of shape (imageH, imageW, 3) pixels and uniform cells of cellH * cellW pixels.
     fillPct% [0, 1] of the cells are != 0. If 'pattern' is an ndarray of shape (patternH, patternW, 3), inserts it into
     the generated image in a random position and also returns a binary feature importance ndarray of shape
     (imageH, imageW) as ground truth explanation for the generated image. """

    totalCells = (imageH / cellH) * (imageW / cellW)
    filledCells = round(totalCells * fillPct)

    # starting pixels (upper left) of each cell
    startingPixelsRow = np.arange(0, imageH, cellH)
    startingPixelsCol = np.arange(0, imageW, cellW)

    # choose random cells to fill
    cellIndexes = zip(rng.choice(startingPixelsRow, size=filledCells), rng.choice(startingPixelsCol, size=filledCells))

    img = np.zeros((imageH, imageW, 3))
    for cellRow, cellCol in cellIndexes:
        # set reg, green and blue for each chosen cell
        img[cellRow:(cellRow+cellH), cellCol:(cellCol+cellW)] = _generate_rgb(rng, colorDev=colorDev)

    if pattern is not None:
        # choose where the pattern goes (upper left corner) and overwrite the image
        patternRow = rng.choice(np.arange(0, imageH - pattern.shape[0], cellH))
        patternCol = rng.choice(np.arange(0, imageW - pattern.shape[1], cellW))
        img[patternRow:(patternRow+pattern.shape[0]), patternCol:(patternCol+pattern.shape[1])] = pattern

        exp = np.zeros((imageH, imageW))
        exp[patternRow:(patternRow + pattern.shape[0]), patternCol:(patternCol + pattern.shape[1])] = binaryPattern
        return img, exp

    return img


def _generate_rgb(rng, colorDev):
    """ Generates a ndarray of shape (3,)representing  RGB color with one of the channels turned to, at least,
    1 - colorDev and the other channels valued at, at most, colorDev. 'colorDev' must be between 0 and 1. """

    # first array position will be turned on, last two turned off
    order = rng.choice(3, size=3, replace=False)
    colors = np.zeros(3)
    colors[order[0]] = 1 - rng.uniform(0, colorDev)
    colors[order[1]] = rng.uniform(0, colorDev)
    colors[order[2]] = rng.uniform(0, colorDev)
    return colors

# ===================================
#       EXPLANATION EVALUATION
# ===================================


def saliency_map_scores(gts, sMaps, metrics=None, binThreshold=.5, gtBackgroundVals='light', average=True) -> float:
    """ Quality metrics for saliency map explanations, where each pixel is considered as a feature.
    Computes different scores of a saliency map explanation w.r.t. its ground truth explanation (a binary mask).
    The saliency map should not be binary. If it is, use the 'binary_mask_scores' method instead.

    # todo support for gts that are not binary
    :param gts: (ndarray) ground truth RGB or binary mask/s. Accepted shapes are
        - (imageH, imageW) A single grayscale mask, where each pixel should be 1 if it is part of the salient class
        and 0 otherwise.
        - (imageH, imageW, 3) A single RGB mask, where pixels that **do not** contain the salient class are all
        either black (all channels set to 0) or white (all channels set to max.).
        - (nSamples, imageH, imageW) Multiple grayscale masks, where for each where, in each image, each pixel should be
        1 if it is part of the salient class and 0 otherwise.
        - (nSamples, imageH, imageW, 3) Multiple RGB masks, where for each image, pixels that **do not** contain the
        salient class are all either black (all channels set to 0) or white (all channels set to max.).
    If the g.t. masks are RGB they will be binarized (see param :code:`gtBackground` to specify the color of the
    pixels that pertain to the non-salient class).
    :param sMaps: (ndarray) grayscale saliency map explanation/s (0-1 normalised). Supported shapes are
        - (imageH, imageW) A single explanation
        - (nSamples, imageH, imageW) Multiple explanations
    :param metrics: (str / array-like of str, default=['auc']) Quality metric/s to compute. Available:
        - 'auc': ROC AUC score. The value of each pixel of each saliency map in :code:`sMaps` is considered as a
        prediction probability of the pixel pertaining to the salient class.
        - 'fscore': F1 Score.
        - 'prec': Precision Score.
        - 'rec': Recall score.
        - 'cs': Cosine Similarity.
    For 'fscore', 'prec', 'rec' and 'cs', the saliency maps in :code:`sMaps` are binarized (see param
    :code:`binThreshold`).
    :param binThreshold: (float in [0, 1]) pixels of images in :code:`sMaps` with a value bigger than this will be set
    to 1 and 0 otherwise when binarizing for the computation of 'fscore', 'prec', 'rec' and 'cs'.
    :param gtBackgroundVals: (str) Only used when provided ground truth explanations are RGB. Color of the background
    of the g.t. masks 'low' if pixels in the mask representing the non-salient class are dark, 'high' otherwise).
    :param average: (bool, default :code:`True`) Used only if :code:`gts` and :code:`sMaps` contain multiple
    observations. Should the computed metrics be averaged across all of the samples?
    :return: (ndarray) specified metric/s in the original order. Can be of shape
        - (n_metrics,) if only one image has been provided in both :code:`gts` and :code:`sMaps` or when both are
        contain multiple observations and :code:`average=True`.
        - (n_metrics, n_samples) if :code:`gts` and :code:`sMaps` contain multiple observations and
        :code:`average=False`. """

    if metrics is None:
        metrics = ['auc']
    elif isinstance(metrics, str):
        metrics = [metrics]

    for metric in metrics:
        if metric not in _AVAILABLE_SALIENCY_MAP_METRICS:
            raise ValueError(f"'{metric}' metric not valid. Use {_AVAILABLE_SALIENCY_MAP_METRICS}")

    # todo treat gts and sMaps depending on their shape

    if len(gts.shape) == 2:
        # 1 binary mask
        if not _array_is_binary(gts):
            pass
    elif len(gts.shape) == 3:
        # either a single rgb mask or a multiple binary masks
        pass
    elif len(gts.shape) == 4:
        # multiple RGB masks
        pass
    else:
        raise ValueError(f'Shape {gts.shape} of ground truth explanations not supported.')

    ##########
    # if not _array_is_binary(gts) and not binarizeGt:
    #     raise ValueError('Ground truth is not binary: binarize before calling.')
    # elif binarizeGt:
    #     if _is_rgb(gts):
    #         gts = _binarize_rgb_mask(gts, bgValue=gtBackgroundVals)
    #     else:
    #         gts[gts != 0] = 1

    sMaps = _normalize_array(sMaps)
    return feature_importance_scores(gts.flatten(), sMaps.flatten(), metrics=metrics, average=average,
                                     binThreshold=binThreshold)


def binary_mask_scores(gt, explanation, metrics=None, **kwargs) -> float:
    # todo revise
    """ Computes metrics for the evaluation of image binary pixel importance explanations.

    :param gt: (ndarray), ground truth pixel importance n x m binary mask.
    :param explanation: (ndarray), image pixel importance n x m binary explanation.
    :param metrics: (str / array-like) 'fscore', 'prec', 'rec', 'auc', 'cs'
    :return: (list) the specified similarity metric/s. """

    return feature_importance_scores(gt.flatten(), explanation.flatten(), metrics=metrics, **kwargs)

# ===================================
#       UTILS
# ===================================

# todo add related utils


def main():
    print('TRANSPARENT MODEL')
    import matplotlib.pyplot as plt

    nSamples = 100
    randomState = 7
    imageH, imageW = 32, 32
    patternH, patternW = 16, 16
    cellHeight, cellWidth = 4, 4
    patternProp = 0.5
    fillPct = 0.4
    colorDev = 0.5

    X, y, _, _, model = gen_image_data(nSamples=nSamples, imageH=imageH, imageW=imageW,
                                       patternH=patternH, patternW=patternW,
                                       cellH=cellHeight, cellW=cellWidth, patternProp=patternProp,
                                       fillPct=fillPct, colorDev=colorDev, randomState=randomState, returnModel=True)

    print(model.predict(X[:5]), y[:5])

    mod = TransparentImageClassifier()
    imgs, _, explanations, pat = gen_image_data(nSamples=4, patternProp=0.5)

    # the model now recognizes the pattern 'pat'
    mod.fit(pat)
    e = mod.explain(imgs)

    _, axs = plt.subplots(4, 2)
    axs[0, 0].imshow(imgs[0])
    axs[0, 0].set_title('Images')
    axs[0, 1].imshow(e[0])
    axs[0, 1].set_title('Explanations')

    axs[1, 0].imshow(imgs[1])
    axs[1, 1].imshow(e[1])

    axs[2, 0].imshow(imgs[2])
    axs[2, 1].imshow(e[2])

    axs[3, 0].imshow(imgs[3])
    axs[3, 1].imshow(e[3])

    plt.show()

    print(mod.predict(imgs))
    print(mod.predict_proba(imgs))

    print(binary_mask_scores(e[0], e[0]))
    print(binary_mask_scores(e[2], e[2]))

    print('SYNTHETIC DATA GENERATION')

    _, axs = plt.subplots(2, 3)
    images, _, e, p = gen_image_data(nSamples=1, patternProp=1, randomState=3)
    axs[0, 0].imshow(p)
    axs[0, 0].set_title('Pattern')
    axs[0, 1].imshow(images[0])
    axs[0, 1].set_title('Generated image')
    axs[0, 2].imshow(e[0])
    axs[0, 2].set_title('Explanation')

    images, y, e, p = gen_image_data(nSamples=1, patternProp=1, randomState=4)
    axs[1, 0].imshow(p)
    axs[1, 1].imshow(images[0])
    axs[1, 2].imshow(e[0])
    plt.show()

    print('EVALUATION')

    from cv2 import erode

    imgpath = '../TAIAOexp_CLI/data/Kahikatea/data/ve_positive/Biospatial_20160702U_1860_01_03.png'

    img = _read_rgb_img(imgpath)

    plt.imshow(img)
    plt.show()

    maskpath = '../TAIAOexp_CLI/data/Kahikatea/expl/ve_positive/Biospatial_20160702U_1860_01_03.png'

    # load ground truth mask and generate perturbed mask for testing
    gtmask = _read_rgb_img(maskpath)

    plt.imshow(gtmask)
    plt.show()

    gtmask = _binarize_rgb_mask(gtmask, bgValue='light')

    plt.imshow(gtmask)
    plt.show()

    # perturb ground truth mask
    testmask = erode(gtmask, np.ones((5, 5), np.uint8))

    plt.imshow(testmask)
    plt.show()

    f1Score, prec, rec, auc, cs = binary_mask_scores(gtmask, testmask, metrics=['fscore', 'prec', 'rec', 'auc', 'cs'])

    print(f'F1Score: {f1Score}, Precision: {prec}, Recall: {rec}, AUC: {auc}, CS: {cs}')


if __name__ == "__main__":
    main()
