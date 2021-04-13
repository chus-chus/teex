from transparentModels.baseClassifier import BaseClassifier


class LinearClassifier(BaseClassifier):
    """ Transparent, linear classifier with feature importances as explanations. """

    def __init__(self):
        super().__init__()

    def fit(self, data, target):
        pass

    def predict(self):
        pass

    def predict_proba(self, obs):
        pass

    def predict_explain(self, obs):
        pass
