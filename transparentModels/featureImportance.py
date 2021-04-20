from transparentModels.baseClassifier import BaseClassifier


class LinearClassifier(BaseClassifier):
    """ Transparent, linear classifier with feature importances as explanations. """

    def __init__(self):
        super().__init__()

    def fit(self, data, target, featureNames=None):
        pass

    def predict(self, obs):
        pass

    def predict_proba(self, obs):
        pass

    def explain(self, obs):
        pass


if __name__ == '__main__':
    pass
