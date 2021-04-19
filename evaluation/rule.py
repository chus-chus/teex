""" Module for the evaluation of decision rule explanations. """

import numpy as np

from evaluation.featureImportance import recall, precision, f_score
from transparentModels.decisionRule import DecisionRule, Statement


from utils.rule import rule_to_feature_importance


def complete_rule_quality(gt: DecisionRule, rule: DecisionRule, eps: float = 0.01) -> float:
    # todo sphinx doc
    """ Computes the complete rule quality between two decision rules. All 'Statements' in both rules must be binary
    (have upper and lower bounds). The metric is defined as the proportion of lower and upper bounds in a rule
    explanation whose that are eps-close to the respective lower and upper bounds (same feature) in the ground truth
    rule explanation amongst those that are != inf. Mathematically:

    :param gt: (DecisionRule) ground truth rule
    :param rule: (DecisionRule)
    :param eps: (float) maximum difference for the bounds to be taken into account in the metric, with precision
                up to 3 decimal places.
    :return: Complete rule quality """

    nBounded = 0
    epsCloseBounds = 0
    for feature in rule.get_features():
        if feature in gt:
            if not gt[feature].binary or not rule[feature].binary:
                raise ValueError('Statements must be binary.')
            # compare upper bounds
            if gt[feature].upperBound != np.inf and rule[feature].upperBound != np.inf:
                nBounded += 1
                if round(abs(gt[feature].upperBound - rule[feature].upperBound), 3) <= eps:
                    epsCloseBounds += 1
            # compare lower bounds
            if gt[feature].lowerBound != -np.inf and rule[feature].lowerBound != -np.inf:
                nBounded += 1
                if round(abs(gt[feature].lowerBound - rule[feature].lowerBound), 3) <= eps:
                    epsCloseBounds += 1
    return epsCloseBounds / nBounded if nBounded != 0 else 0


def rule_scores(gt: DecisionRule, rule: DecisionRule, allFeatures, metrics=None, beta=1, **kwargs) -> float:
    """ Fscore, precision and recall scores for Decision Rules (interpreted as binary feature importance vectors).

    :param gt: ground truth decision rule.
    :param rule: approximated decision rule.
    :param allFeatures: (array-like) names of all of the relevant features.
    :param metrics: (array-like, default = ['fscore']) metris to compute.
    :param beta: beta value used when computing the fscore.
    :return: (list) specified metrics.
    """
    if metrics is None:
        metrics = ['fscore']

    gt = rule_to_feature_importance(gt, allFeatures)
    rule = rule_to_feature_importance(rule, allFeatures)

    res = []
    for metric in metrics:
        if metric == 'fscore':
            res.append(f_score(gt, rule, beta=beta, **kwargs))
        elif metric == 'prec':
            res.append(precision(gt, rule, **kwargs))
        elif metric == 'rec':
            res.append(recall(gt, rule, **kwargs))
        else:
            raise ValueError(f"Metric not valid. Use: {['fscore', 'prec', 'rec']}")
    return res


if __name__ == '__main__':
    print('COMPLETE RULE QUALITY')
    c1 = Statement('a', binary=True, lowerBound=2, upperBound=1)
    c2 = Statement('a', binary=True, lowerBound=2.2, upperBound=1.1)
    r1 = DecisionRule([c1])
    r2 = DecisionRule([c2])

    print(complete_rule_quality(r1, r2, eps=0.01))
    print(complete_rule_quality(r1, r2, eps=0.1))
    print(complete_rule_quality(r1, r2, eps=0.2))

    c3 = Statement('b', binary=True, lowerBound=2.2, upperBound=1.1)
    r3 = DecisionRule([c3])

    print(complete_rule_quality(r1, r3, eps=0.2))

    print('The metric does not take the absence of a feature into account:')
    r3 = DecisionRule([c3, c2])
    print(complete_rule_quality(r1, r3, eps=0.2))

    print('F1SCORE')
    features = ['a', 'b']
    mets = ['fscore', 'prec', 'rec']
    print('r3:', rule_to_feature_importance(r3, features))
    print('r1:', rule_to_feature_importance(r1, features))

    fscore, precision, recall = rule_scores(r3, r1, features, mets)

    print(fscore)
    print('PRECISION')
    print(precision)
    print('RECALL')
    print(recall)

