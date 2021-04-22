import numpy as np

from transparentModels.baseClassifier import BaseClassifier


class ImageClassifier(BaseClassifier):
    """ Transparent, pixel-based classifier with pixel (features) importances as explanations. Predicts the
    class of the images based on whether they contain a certain specified pattern or not. Class 1 if they contain
    the pattern, 0 otherwise. To be trained only a pattern needs to be fed.

    IMPORTANT NOTE ON GENERATING G.T. EXPLANATIONS: the ground truth explanations are automatically generated with
    the 'dataGen.gen_image_data' method, so using this class to generate explanations is not recommended,
    as it is much less efficient. """

    def __init__(self):
        super().__init__()
        self.pattern = None
        self.pattH = None
        self.pattW = None
        self._binaryPattern = None

    def fit(self, pattern):
        self.pattern = pattern
        self.pattH = pattern.shape[0]
        self.pattW = pattern.shape[1]
        self._binaryPattern = np.squeeze(np.where(np.delete(pattern, (0, 1), 2) != 0, 1, 0))

    def predict(self, obs):
        """ Predicts the class for each observation.
            :param obs: array of n images as ndarrays.
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
        if specified. """

        hasPat = False
        indices = (0, 0)
        for row in range(len(image[0]) - self.pattH):
            for col in range(len(image[1]) - self.pattW):
                hasPatRow = False
                # instead of checking all of the pattern at once, check by rows for efficiency
                # not expading another loop for code simplicity
                for pattRow in range(self.pattH):
                    hasPatRow = (image[row+pattRow, col:(col+self.pattW)] == self.pattern[pattRow]).all()
                    if not hasPatRow:
                        break
                if hasPatRow:
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

    mod = ImageClassifier()
    imgs, explanations, pat = gen_image_data(nSamples=4, patternProp=0.5)

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
