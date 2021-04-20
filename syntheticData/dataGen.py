""" Module for generation of synthetic datasets and their ground truth explanations. """
import random

import numpy as np
from sklearn.datasets import make_classification

from transparentModels.decisionRule import RuleClassifier
from utils.rule import rule_to_feature_importance


def gen_tabular_data(nSamples: int = 1000, nClasses: int = 2, nFeatures: int = 3, nInformative: int = None,
                     nRedundant: int = None, randomState: int = 888, explanations='rule', returnModel=False):
    """ Generate synthetic classification tabular data with ground truth explanations as feature importance vectors.

    :param nSamples: number of samples in the data
    :param nClasses: numer of classes in the data
    :param nFeatures: total number of features
    :param nInformative: number of informative features
    :param nRedundant: number of redundant features
    :param randomState: random seed
    :param explanations: how the explanations are generated. If None, no explanations are computed.
                         Available: 'rule' (vectors will be binary), 'linear' (vectors will be real-valued)
    :param returnModel: should the model used for the generation of the explanations be returned?
                        (only if 'explanations' != None)
    :return: arrays X, y, explanations (optional), model (optional)
    """
    if nInformative is None and nRedundant is not None:
        nInformative = nFeatures - nRedundant
    elif nRedundant is None and nInformative is not None:
        nRedundant = nFeatures - nInformative
    else:
        nInformative = nFeatures
        nRedundant = 0
    data, targets = make_classification(n_samples=nSamples, n_classes=nClasses, n_features=nFeatures,
                                        n_informative=nInformative, n_redundant=nRedundant, random_state=randomState)
    # todo add randomness

    if explanations is None:
        if returnModel:
            raise ValueError('Cannot return model without generating explanations.')
        else:
            return data, targets

    if explanations == 'rule':
        classifier = RuleClassifier(random_state=randomState)
        classifier.fit(data, targets)
        explanations = rule_to_feature_importance(classifier.explain(data), classifier.featureNames)
    elif explanations == 'linear':
        raise NotImplementedError
    else:
        raise ValueError(f"Explanation method not valid. Use {['rule', 'linear']}")

    if returnModel:
        return data, targets, explanations, classifier
    else:
        return data, targets, explanations


def gen_image_data(nSamples=1000, imageH=32, imageW=32, patternH=16, patternW=16, cellH=4, cellW=4, patternProp=0.5,
                   fillPct=0.4, colorDev=0.1, randomState=888, returnModel=False):
    if imageH % patternH != 0 or imageW % patternW != 0 or imageH % cellH != 0 or imageW % cellW != 0:
        raise ValueError('Image height and widht not multiple of cell or pattern dimensions.')
    if imageH < patternH or imageH < cellH or imageW < patternW or imageW < cellW or patternH < cellH or \
            patternW < cellW:
        raise ValueError('Cells should be smaller than patterns and patterns than image size.')

    random.seed(randomState)
    rng = np.random.default_rng(randomState)

    nWithPattern = round(nSamples * patternProp)
    pattern = _generate_image(imageH=patternH, imageW=patternW, cellH=cellH, cellW=cellW, fillPct=fillPct, rng=rng,
                              colorDev=colorDev, pattern=None)

    # transform pattern into a 2d array and then set the channel to 1 if the pixel had any intensity at all (if a cell
    # was part of the pattern it will have at least some intensity). Squeeze it so it has shape patH x patW.
    binaryPattern = np.squeeze(np.where(np.delete(pattern, (0, 1), 2) != 0, 1, 0))

    imgs = []
    for _ in range(nWithPattern):
        image, explanation = _generate_image(imageH=imageH, imageW=imageW, cellH=cellH, cellW=cellW, fillPct=fillPct,
                                             rng=rng, colorDev=colorDev, pattern=pattern, binaryPattern=binaryPattern)
        imgs.append((image, explanation))
    for _ in range(nSamples - nWithPattern):
        image = _generate_image(imageH=imageH, imageW=imageW, cellH=cellH, cellW=cellW, fillPct=fillPct,
                                rng=rng, colorDev=colorDev, pattern=None)
        imgs.append((image, None))

    random.shuffle(imgs)
    imgs, exps = zip(*imgs)

    if returnModel:
        # todo instantiate model
        # todo return is not correct
        return imgs, exps, pattern, 0
    else:
        return imgs, exps, pattern


def _generate_image(imageH, imageW, cellH, cellW, fillPct, rng, colorDev, pattern=None, binaryPattern=None):
    """ Generates RGB image with imageH * imageW pixels and uniform cells of cellH * cellW pixels.
     fillPct% [0, 1] of the cells are != 0. If 'pattern' is not None, returns a binary feature
     importance array as ground truth explanation too. """

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
        # todo add random rotations to the pattern?
        # choose where the pattern goes (upper left corner) and overwrite the image
        patternRow = rng.choice(np.arange(0, imageH - pattern.shape[0], cellH))
        patternCol = rng.choice(np.arange(0, imageW - pattern.shape[1], cellW))
        img[patternRow:(patternRow+pattern.shape[0]), patternCol:(patternCol+pattern.shape[1])] = pattern

        exp = np.zeros((imageH, imageW))
        exp[patternRow:(patternRow + pattern.shape[0]), patternCol:(patternCol + pattern.shape[1])] = binaryPattern
        return img, exp

    return img


def _generate_rgb(rng, colorDev):
    """ Generates an RGB color with one of the channels turned to, at least, 1 - colorDev and
     the other channels valued at, at most, colorDev. 'colorDev' must be between 0 and 1. """
    # first array position will be turned on, last two turned off
    order = rng.choice(3, size=3, replace=False)
    colors = np.zeros(3)
    colors[order[0]] = 1 - rng.uniform(0, colorDev)
    colors[order[1]] = rng.uniform(0, colorDev)
    colors[order[2]] = rng.uniform(0, colorDev)
    return colors


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3)
    images, explanations, pattern = gen_image_data(nSamples=1, patternProp=1)
    axs[0].imshow(pattern)
    axs[0].set_title('pattern')
    axs[1].imshow(images[0])
    axs[1].set_title('image')
    axs[2].imshow(explanations[0])
    axs[2].set_title('explanation')
    plt.show()

    print('AUTOMATICALLY GENERATING RULE EXPLANATIONS.')
    X, y = gen_tabular_data(explanations=None)
    print('\n', X[1:3], y[1:3])
    X, y, ex = gen_tabular_data()
    print('\n', X[1:3], y[1:3], ex[1:3])
    X, y, ex, model = gen_tabular_data(returnModel=True)
    print('\n', X[1:3], y[1:3], ex[1:3], model)

    print('\nONE CAN ALSO EXTRACT THE ORIGINAL RULE EXPLANATIONS FROM THE MODEL:')
    explan = model.explain(X[1:3])
    for e in explan:
        print(type(e), e)
