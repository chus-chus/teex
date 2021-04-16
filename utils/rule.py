""" Rule utils """

from transparentModels.decisionRule import DecisionRule


def rule_to_feature_importance(rule: DecisionRule, allFeatures) -> list:
    """  Converts a DecisionRule object to a feature importance vector
    :param rule: (DecisionRule) Rule to convert
    :param allFeatures: (array-like) List with m features (same as the rule features) whose order the returned array
    will follow
    :return: binary array of length m where a 1 indicates that the feature appears in the rule
    """
    return [1 if feature in rule else 0 for feature in allFeatures]
