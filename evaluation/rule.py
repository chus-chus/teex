""" Module for the evaluation of decision rule explanations. """

import numpy as np

from evaluation.featureImportance import feature_importance_scores
from transparentModels.decisionRule import DecisionRule, Statement


from utils.rule import rule_to_feature_importance

# todo filter kwargs in *_scores functions


def complete_rule_quality(gt: DecisionRule, rule: DecisionRule, eps: float = 0.1) -> float:
    # todo sphinx doc
    """ Computes the complete rule quality (crq) between two decision rules. All 'Statements' in both rules must be
    binary (have upper and lower bounds). The metric is defined as the proportion of lower and upper bounds in a rule
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


def rule_scores(gt: DecisionRule, rule: DecisionRule, allFeatures, metrics=None, **kwargs) -> float:
    """ Fscore, precision and recall scores for Decision Rules (interpreted as binary feature importance vectors).

    :param gt: ground truth decision rule.
    :param rule: approximated decision rule.
    :param allFeatures: (array-like) names of all of the relevant features.
    :param metrics: (list, default = ['fscore']) metris to compute. Available: ['fscore', 'prec', 'rec', 'crq',
                                                 'auc', 'cs']
    :return: (list) specified metric/s.
    """
    isArrayLike = isinstance(metrics, (list, np.ndarray, tuple))
    crq = None
    if metrics is not None:
        if metrics == 'crq':
            return complete_rule_quality(gt, rule, **kwargs)
        elif isArrayLike and 'crq' in metrics:
            crq = complete_rule_quality(gt, rule, **kwargs)
            crqIndex = metrics.index('crq')
            del metrics[metrics.index('crq')]

    binaryGt = rule_to_feature_importance(gt, allFeatures)
    binaryRule = rule_to_feature_importance(rule, allFeatures)

    res = feature_importance_scores(binaryGt, binaryRule, metrics=metrics, **kwargs)

    if crq is not None:
        # noinspection PyUnboundLocalVariable
        res.insert(crqIndex, crq)
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

    print(r1)
    print(r3)
    print(rule_scores(r3, r1, features, ['crq', 'prec', 'rec', 'fscore', 'cs']))
