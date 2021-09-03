""" Module for evaluation of decision rule explanations. """

import numpy as np

from teex._utils._errors import MetricNotAvailableError
from teex.decisionRule.data import DecisionRule, rule_to_feature_importance, Statement
from teex.featureImportance.eval import feature_importance_scores

_AVAILABLE_DECISION_RULE_METRICS = {'fscore', 'prec', 'rec', 'cs', 'auc', 'crq'}


def complete_rule_quality(gts: DecisionRule, rules: DecisionRule, eps: float = 0.1) -> float:
    """ Computes the complete rule quality (crq) between two decision rules. All 'Statements' in both rules must be
    binary (have upper and lower bounds). The metric is defined as the proportion of lower and upper bounds in a rule
    explanation that are eps-close to the respective lower and upper bounds (same feature) in the ground truth
    rule explanation amongst those that are :math:`\\neq \\infty`. Mathematically, given two rules :math:`e, \\tilde{e}`
    and a similarity threshold :math:`\\varepsilon`, the quality of :math:`e` with respect to :math:`\\tilde{e}` is:

    .. math::

        q(e, \\tilde{e}) = \\frac{1}{N_{\\not\\infty}} \\sum_{i=1}^{|e|}{\\delta_{\\varepsilon}(e_i, \\tilde e_i)},

    where

    .. math::

        \\delta_{\\varepsilon}(e_i, \\tilde{e}_i) =
                \\begin{cases}
                    1 & \\text{if } |e_i - \\tilde{e}_i| \\leq \\varepsilon \\wedge |e_i| \\neq \\infty \\wedge
                        |\\tilde{e}_i| \\neq \\infty, \\newline
                    0 & \\text{otherwise}
                \\end{cases}

    Where :math:`N_{\\not \\infty}` is the number of lower and upper bounds that are different from :math:`\\infty` in
    both :math:`e` and :math:`\\tilde e`. More about this metric can be found in [Evaluating local explanation methods
    on ground truth, Riccardo Guidotti, 2021].

    :Example:

    >>> c1 = Statement('a',lowB=2,upperB=1)
    >>> c2 = Statement('a',lowB=2.2,upperB=1.1)
    >>> r1 = DecisionRule([c1])
    >>> r2 = DecisionRule([c2])  # both rules contain the feature 'a'

    >>> print(complete_rule_quality(r1, r2, eps=0.01))
    >>> print(complete_rule_quality(r1, r2, eps=0.1))
    >>> print(complete_rule_quality(r1, r2, eps=0.2))

    >>> c3 = Statement('b',lowB=2.2,upperB=1.1)
    >>> r3 = DecisionRule([c3])

    >>> print(complete_rule_quality(r1, r3, eps=0.2))

    >>> # The metric does not take the absence of a feature into account
    >>> r3 = DecisionRule([c3, c2])
    >>> print(complete_rule_quality(r1, r3, eps=0.2))

    :param gts: (DecisionRule or array-like of DecisionRules) ground truth rule w.r.t. which to compute the quality
    :param rules: (DecisionRule or array-like of DecisionRules) rule to compute the quality for
    :param eps: (float) threshold :math:`\\varepsilon` for the bounds to be taken into account in the metric, with
        precision up to 3 decimal places.
    :return: (float or ndarray of shape (n_samples,)) Complete rule quality. """

    if isinstance(gts, DecisionRule) and isinstance(rules, DecisionRule):
        gts, rules = [gts], [rules]
    if not isinstance(gts, (list, np.ndarray, tuple)) or not isinstance(rules, (list, np.ndarray, tuple)):
        raise ValueError('Rules format not supported.')

    res = []
    for gt, rule in zip(gts, rules):
        nBounded = 0
        epsCloseBounds = 0
        for feature in rule.get_features():
            if feature in gt:
                if not gt[feature].binary or not rule[feature].binary:
                    raise ValueError('Statements must be binary.')
                # compare upper bounds
                if gt[feature].upperB != np.inf and rule[feature].upperB != np.inf:
                    nBounded += 1
                    if round(abs(gt[feature].upperB - rule[feature].upperB), 3) <= eps:
                        epsCloseBounds += 1
                # compare lower bounds
                if gt[feature].lowB != -np.inf and rule[feature].lowB != -np.inf:
                    nBounded += 1
                    if round(abs(gt[feature].lowB - rule[feature].lowB), 3) <= eps:
                        epsCloseBounds += 1
        res.append(epsCloseBounds / nBounded if nBounded != 0 else 0)

    return res[0] if len(res) == 1 else np.array(res).astype(np.float32)


def rule_scores(gts: DecisionRule, rules: DecisionRule, allFeatures, metrics=None, average=True,
                crqParams=None) -> float:
    """ Quality metrics for :class:`teex.decisionRule.data.DecisionRule` objects.

    :param gts: (DecisionRule or array-like of DecisionRules) ground truth decision rule/s.
    :param rules: (DecisionRule or array-like of DecisionRules) approximated decision rule/s.
    :param allFeatures: (array-like) names of all of the relevant features (i.e. :code:`featureNames`
        of :class:`teex.decisionRule.data.SenecaDR` object.)
    :param metrics: (array-like of str, default :code:`['fscore']`) metrics to compute. Available:

        - 'fscore': Computes the F1 Score between the ground truths and the predicted vectors.
        - 'prec': Computes the Precision Score between the ground truths and the predicted vectors.
        - 'rec': Computes the Recall Score between the ground truths and the predicted vectors.
        - 'crq': Computes the Complete Rule Quality of :code:`rule` w.r.t. :code:`gt`.
        - 'auc': Computes the ROC AUC Score between the two rules.
        - 'cs': Computes the Cosine Similarity between the two rules.

        Note that for 'fscore', 'prec', 'rec', 'auc' and 'cs' the rules are transformed to binary vectors where there
        is one entry per possible feature and that entry contains a 1 if the feature is present in the rule, otherwise
        0.
    :param average: (bool, default :code:`True`) Used only if :code:`gts` and :code:`rule` are array-like. Should the
        computed metrics be averaged across all of the samples?
    :param dict crqParams: Extra parameters complete rule quality.
    :return: (ndarray) specified metric/s in the original order. Can be of shape

        - (n_metrics,) if only one DecisionRule has been provided in both :code:`gts` and :code:`rules` or when both are
          array-like and :code:`average=True`.
        - (n_metrics, n_samples) if :code:`gts` and :code:`rules` are array-like and :code:`average=False`. """

    isArrayLike = isinstance(metrics, (list, np.ndarray, tuple))
    if not isArrayLike:
        if metrics not in _AVAILABLE_DECISION_RULE_METRICS:
            raise MetricNotAvailableError(metrics)
    else:
        for metric in metrics:
            if metric not in _AVAILABLE_DECISION_RULE_METRICS:
                raise MetricNotAvailableError(metric)

    crq, crqIndex = None, None
    crqParams = dict() if crqParams is None else crqParams
    if metrics is not None:
        if metrics == 'crq':
            return complete_rule_quality(gts, rules, **crqParams)
        elif isArrayLike and 'crq' in metrics:
            crq = complete_rule_quality(gts, rules, **crqParams)
            crqIndex = metrics.index('crq')
            del metrics[crqIndex]

    binaryGts = rule_to_feature_importance(gts, allFeatures)
    binaryRules = rule_to_feature_importance(rules, allFeatures)

    res = feature_importance_scores(binaryGts, binaryRules, metrics=metrics, average=average)

    if crq is not None:
        if average:
            crq = np.mean(crq)
        if isinstance(crq, np.ndarray):
            # multiple observations, insert as a column
            res = np.insert(res, crqIndex, crq, axis=1)
        elif isinstance(crq, (float, int, np.float32)):
            # one observation
            res = np.insert(res, crqIndex, crq, axis=0)

        if isArrayLike:
            metrics.insert(crqIndex, 'crq')
    return res
