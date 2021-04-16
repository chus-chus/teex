""" Module for the evaluation of decision rule explanations. """

import numpy as np

from transparentModels.decisionRule import DecisionRule

# todo functions for recall, precision and fscore of rules


def complete_rule_quality(gt: DecisionRule, rule: DecisionRule, eps: float = 0.01) -> float:
    # todo sphinx doc
    """ Computes the complete rule quality between two decision rules. All 'Statements' in both rules must be binary
    (have upper and lower bounds). The metric is defined as the proportion of lower and upper bounds in a rule
    explanation whose that are eps-close to the respective lower and upper bounds (same feature) in the ground truth
    rule explanation amongst those that are != inf. Mathematically:

    :param gt: (DecisionRule) ground truth rule
    :param rule: (DecisionRule)
    :param eps: (float) maximum difference for the bounds to be taken into account in the metric.
    :return: Complete rule quality """

    nBounded = 0
    epsCloseBounds = 0
    for feature in rule.get_features():
        if feature in gt:
            if not gt[feature].binary or not rule[feature].binary:
                raise ValueError('Statements must be binary.')
            if gt[feature].upperBound != np.inf and rule[feature].upperBound != -np.inf:
                nBounded += 1
                if abs(gt[feature].upperBound - rule[feature].upperBound) <= eps:
                    epsCloseBounds += 1
    return epsCloseBounds / nBounded if nBounded != 0 else 0


if __name__ == '__main__':
    pass
