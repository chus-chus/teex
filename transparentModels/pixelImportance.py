import numpy as np

from transparentModels.baseClassifier import BaseClassifier


class ImageClassifier(BaseClassifier):
    """ Transparent, pixel-based classifier with pixel (features) importances as explanations. Predicts the
    class of the images based on whether they contain a certain specified pattern or not. Class 1 if they contain
    the pattern, 0 otherwise. To be trained only a pattern needs to be fed.

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


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from syntheticData.image import gen_image_data
    from evaluation.image import binary_mask_scores

    nSamples = 100
    randomState = 8
    imageH, imageW = 32, 32
    patternH, patternW = 16, 16
    cellHeight, cellWidth = 4, 4
    patternProp = 0.5
    fillPct = 0.4
    colorDev = 0.1

    X, y, _, _, model = gen_image_data(nSamples=100, imageH=imageH, imageW=imageW,
                                       patternH=patternH, patternW=patternW,
                                       cellH=cellHeight, cellW=cellWidth, patternProp=patternProp,
                                       fillPct=fillPct, colorDev=0.5, randomState=7, returnModel=True)

    print(model.predict(X[:5]), y[:5])

    mod = ImageClassifier()
    imgs, y, explanations, pat = gen_image_data(nSamples=4, patternProp=0.5)

    # the model now recognizes the pattern 'pat'
    mod.fit(pat)
    e = mod.explain(imgs)

    fig, axs = plt.subplots(4, 2)
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
