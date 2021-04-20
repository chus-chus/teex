import abc


class BaseClassifier(object):
    def __init__(self):
        self.featureNames = None
        self.classNames = None
        self.classValues = None

    @abc.abstractmethod
    def fit(self, data, target):
        """ Fits the model to the data """
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
