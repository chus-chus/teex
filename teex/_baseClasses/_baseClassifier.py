import abc

""" Base classifier used for the seneca data generation methods. """


class _BaseClassifier(object):
    def __init__(self):
        self.featureNames = None

    @abc.abstractmethod
    def fit(self, *args, **kwargs):
        """ Fits the model. """
        pass

    @abc.abstractmethod
    def predict_proba(self, obs):
        """ Returns predicted probabilities for each class label and observation. """
        pass

    @abc.abstractmethod
    def predict(self, obs):
        """ Returns predicted label for each observation. """
        pass

    @abc.abstractmethod
    def explain(self, obs):
        """ Returns explanation for each observation. """
        pass
