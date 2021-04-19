""" Rule utils """
import numpy as np

from transparentModels.decisionRule import DecisionRule


def rule_to_feature_importance(rules, allFeatures) -> list:
    """  Converts a DecisionRule object to a feature importance vector
    :param rules: (DecisionRule or 1*r array-like) Rule/s to convert
    :param allFeatures: (array-like) List with m features (same as the rule features) whose order the returned array
    will follow
    :return: binary array (list if > 1 rule) of length m where a 1 indicates that the feature appears in the rule
    """
    if isinstance(rules, (list, np.ndarray, tuple)):
        res = []
        for rule in rules:
            res.append([1 if feature in rule else 0 for feature in allFeatures])
    else:
        res = [1 if feature in rules else 0 for feature in allFeatures]
    return res
