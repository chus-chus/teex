import abc


class BaseClassifier(object):
    def __init__(self):
        self.featureNames = None
        self.classNames = None
        self.classValues = None

    @abc.abstractmethod
    def fit(self, data, target):
        pass

    @abc.abstractmethod
    def predict_proba(self, obs):
        pass

    @abc.abstractmethod
    def predict(self, obs):
        pass

    @abc.abstractmethod
    def predict_explain(self, obs):
        """ Returns predicted label with probability and explanation. """
        pass
