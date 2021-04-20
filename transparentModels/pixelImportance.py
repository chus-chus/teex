from transparentModels.baseClassifier import BaseClassifier


class ImageClassifier(BaseClassifier):
    """ Transparent, pixel-based classifier with pixel (features) importances as explanations. """

    def __init__(self):
        super().__init__()

    def fit(self, data, pattern):
        pass

    def predict(self, obs):
        pass

    def predict_proba(self, obs):
        pass

    def explain(self, obs):
        pass
