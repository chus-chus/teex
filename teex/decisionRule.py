""" The rule module. Contains all of the methods regarding this datatype: synthetic data generation, evaluation of
explanations and other utils. """

import numpy as np
from numpy import ndarray
from sklearn.tree import DecisionTreeClassifier

from teex.featureImportance import feature_importance_scores

# noinspection PyProtectedMember
from teex._baseClasses._baseClassifier import _BaseClassifier
# noinspection PyProtectedMember
from teex._utils._misc import _generate_feature_names

_AVAILABLE_DECISION_RULE_METRICS = {'fscore', 'prec', 'rec', 'cs', 'auc', 'crq'}
_AVAILABLE_DECISION_RULE_GEN_METHODS = {'seneca'}
_AVAILABLE_DECISION_RULE_DATASETS = {}

_VALID_OPERATORS = {'=', '!=', '>', '<', '>=', '<='}
# operators for statements of shape: value1 <opLower> feature <opUpper> value2
_BINARY_OPERATORS = {'<', '<='}


# ===================================
#       TRANSPARENT MODEL
# ===================================

class Statement(object):
    f""" Class representing the atomic structure of a rule. A Statement follows the structure of 'feature' 
    <operator> 'value'. It can also be binary, like so: value1 <operatorLower> feature <operatorHigher> value2 Valid 
    operators are {_VALID_OPERATORS} or {_BINARY_OPERATORS} in the case of a binary statement. """

    def __init__(self, feature, binary=False, lowOp='<', lowerBound=-np.inf, upperOp='<=', upperBound=np.inf, op='=',
                 val=np.inf):
        """
        :param feature: (str) name of the feature for the Statement
        :param binary: (bool) is the statement binary?
        :param lowOp: (str) Operator for the lower bound (if binary)
        :param lowerBound: Value of the upper bound (if binary)
        :param upperOp: (str) Operator for the upper bound (if binary)
        :param upperBound: Value of the lower bound (if binary)
        :param op: (str) Operator for the statement (if not binary)
        :param val: Value for the statement (if not binary)
        """
        self.feature = feature
        self.binary = binary
        if binary is True:
            self._check_binary_operators(lowOp, upperOp)
            self.lowerOperator = lowOp
            self.upperOperator = upperOp
            self.lowerBound = lowerBound
            self.upperBound = upperBound
        else:
            self._check_operator(op)
            self.operator = op
            self.value = val

    def __eq__(self, other):
        if self.binary:
            return self.feature == other.feature and self.lowerOperator == other.lowerBound and \
                   self.upperOperator == other.upperBound and self.lowerBound == other.lowerBound and \
                   self.upperBound == other.upperBound
        else:
            return self.feature == other.feature and self.operator == other.operator and self.value == other.value

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        if self.binary:
            if self.lowerBound != -np.inf and self.upperBound != np.inf:
                return f"{self.lowerBound} {self.lowerOperator} '{self.feature}' {self.upperOperator} {self.upperBound}"
            elif self.lowerBound != -np.inf:
                return f"{self.lowerBound} {self.lowerOperator} '{self.feature}'"
            elif self.upperBound != np.inf:
                return f"'{self.feature}' {self.upperOperator} {self.upperBound}"
            else:
                return f"'{self.feature}' not bounded"
        else:
            if self.value != np.inf:
                return f"'{self.feature}' {self.operator} {self.value}"
            else:
                return f"'{self.feature}' not bounded"

    def __hash__(self):
        return hash(str(self))

    @staticmethod
    def _check_operator(op):
        if op not in _VALID_OPERATORS:
            raise ValueError(f"Operator '{op}' not valid. Choose from {_VALID_OPERATORS}")

    @staticmethod
    def _check_binary_operators(lowerOp, upperOp):
        if lowerOp not in _BINARY_OPERATORS or upperOp not in _BINARY_OPERATORS:
            raise ValueError(f'Operator not valid, must be in {_BINARY_OPERATORS} for a binary statement.')


class DecisionRule(object):
    """ A conjunction of statements as conditions that imply a result. Internally, the rule is represented as a
    dictionary of statements with the feature names as keys. A feature cannot have more than one Statement (
    Statements can be binary). This class is capable of adapting previous Statements depending on new Statements that
    are added to it with the upsert method (see `.upsert_condition` method).
    """

    # todo add renaming of features
    def __init__(self, statements=None, result=None):
        """
        :param statements: (list-like) Statement objects.
        :param result: Statement object.
        """
        self.statements = {}
        if statements is not None:
            # if array-like, convert it into a dict
            if isinstance(statements, (np.ndarray, tuple, list)):
                for statement in statements:
                    if not isinstance(statement, Statement):
                        raise ValueError('Statements are not instances of the Statement class.')
                    elif statement.feature in self.statements:
                        raise ValueError('Only one rule per feature is allowed.')
                    else:
                        self.statements[statement.feature] = statement
            else:
                raise ValueError('Statements not valid.')

        if result is not None and not isinstance(result, Statement):
            raise ValueError('Result must be an Statement.')

        self.result = result

    def __str__(self):
        return f"IF {', '.join([str(statement) for statement in self.statements.values()])} THEN {self.result}"

    def __len__(self):
        return len(self.statements)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return self.statements == other.statements and self.result == other.result

    def __ne__(self, other):
        return not self.__eq__(other)

    def __getitem__(self, feature):
        return self.statements[feature]

    def __contains__(self, feature: str):
        """ Check if a Statement related to a feature is in the rule. """
        return True if feature in self.statements else False

    def insert_condition(self, statement: Statement):
        """ Add Condition inplace to the conjunction.

        :param statement: Statement object
        """
        if statement.feature in self.statements:
            raise ValueError('A condition for this feature already exists.')
        else:
            self.statements[statement.feature] = statement

    def upsert_condition(self, statement: Statement, updateOperators: bool = True):
        """ If a statement already exists within the rule, updates its bounds (replacing or defining them) and its
        operators if specified. If not, inserts the statement as a new condition. If an existing condition is of
        different type (binary / non-binary) as the new condition, the update fails. A bound update is only performed
        if the new bound/s != np.inf or -np.inf.

        :param statement: Statement object
        :param updateOperators: Should the operators be updated too? """

        if statement.feature not in self.statements:
            self.statements[statement.feature] = statement
        else:
            if statement.binary and self.statements[statement.feature].binary:
                if statement.upperBound != np.inf:
                    self.statements[statement.feature].upperBound = statement.upperBound
                if statement.lowerBound != -np.inf:
                    self.statements[statement.feature].lowerBound = statement.lowerBound
                if updateOperators:
                    self.statements[statement.feature].upperOperator = statement.upperOperator
                    self.statements[statement.feature].lowerOperator = statement.lowerOperator
            elif not statement.binary and not self.statements[statement.feature].binary:
                self.statements[statement.feature].value = statement.value
                if updateOperators:
                    self.statements[statement.feature].operator = statement.operator
            else:
                prevStatType = 'binary' if self.statements[statement.feature].binary else 'non-binary'
                newStatType = 'binary' if statement.binary else 'non-binary'
                raise ValueError(f'Cannot update {prevStatType} statement with new {newStatType} statement.')

    def get_features(self) -> list:
        """ Returns list with the features present in the rule conditions. """
        return list(self.statements.keys())

    def set_result(self, result):
        self.result = result


class TransparentRuleClassifier(_BaseClassifier):
    """ Transparent, rule-based classifier with decision rules as explanations. For each prediction, the associated
    ground truth explanation is available with the :code:`.explain` method. Follows the sklean API. Presented in
    [Evaluating local explanation methods on ground truth, Riccardo Guidotti, 2021]. """

    def __init__(self, **kwargs):
        super().__init__()
        self.model = DecisionTreeClassifier(**kwargs)
        # dict. for each tree node, contains its learned condition as a "Statement" (None if node is a leaf)
        self._nodeStatements = None

    def fit(self, data, target, featureNames=None):
        """ Fits the classifier and automatically parses the learned tree structure into statements.

        :param data: (array-like) of shape (n_samples, n_features) The training input samples. Internally, it will be
            converted to dtype=np.float32.
        :param target: (array-like of shape (n_samples,) or (n_samples, n_outputs)) The target values (class labels) as
            integers or strings.
        :param featureNames: (str) todo
        """
        self.model.fit(data, target)

        if featureNames is None:
            self.featureNames = [str(num) for num in range(data.shape[1])]
        else:
            self.featureNames = featureNames

        self._parse_tree_structure()

    def predict(self, obs):
        """ Predicts the class for each observation.
        :param obs: (array-like) of n observations with m features and shape (n, m)
        :return: array of n predicted labels
        """
        return self.model.predict(obs)

    def predict_proba(self, obs):
        """ Predicts probability that each observation belongs to each of the c classes.
        :param obs: array of n observations with m features and shape (n, m)
        :return: array of n probability tuples of length c
        """
        return self.model.predict_proba(obs)

    def explain(self, obs):
        """  Explain observations' predictions with decision rules.

        :param obs: array of n observations with m features and shape (n, m)
        :return: list with n :code:`DecisionRule` objects """
        nodeIndicators = self.model.decision_path(obs)  # id's of the nodes for which each observation passes through
        rules = []
        # for each sample, retrieve its path and navigate it, looking at the precomputed decision splits
        for sampleId in range(len(obs)):
            nodePath = nodeIndicators.indices[nodeIndicators.indptr[sampleId]:nodeIndicators.indptr[sampleId+1]]
            rule = DecisionRule()
            for nodeId in nodePath:
                # node is not a leaf if None
                if self._nodeStatements[nodeId] is not None:
                    feature = self._nodeStatements[nodeId].feature
                    threshold = self._nodeStatements[nodeId].value
                    statement = Statement(self.featureNames[feature], binary=True, lowOp='<', upperOp='<=')
                    # define bounds. Remember that the tree splits are of the form: feature '<=' value
                    if obs[sampleId][feature] <= threshold:
                        statement.upperBound = round(threshold, 3)
                    else:
                        statement.lowerBound = round(threshold, 3)
                    # we create binary statements and will update the bounds as we traverse the tree's decision
                    rule.upsert_condition(statement, updateOperators=False)
            rule.set_result(Statement('Class', op='=', val=self.predict(obs[sampleId].reshape(1, -1))[0]))
            rules.append(rule)
        return rules

    def _parse_tree_structure(self):
        """ Precomputes the learned tree splits and stores them as unary :code:`Statement` objects. """

        nodeFeatures = self.model.tree_.feature
        nodeThresholds = self.model.tree_.threshold
        childrenRight = self.model.tree_.children_right
        childrenLeft = self.model.tree_.children_left

        self._nodeStatements = {nodeId: None for nodeId in range(self.model.tree_.node_count)}

        # root node
        nodeStack = [0]
        while len(nodeStack) > 0:
            nodeId = nodeStack.pop()
            if childrenRight[nodeId] != childrenLeft[nodeId]:
                # nodeId is a split node; add child nodes to the stack and parse its decision split into a statement
                nodeStack.append(childrenRight[nodeId])
                nodeStack.append(childrenLeft[nodeId])
                # all split nodes check with the '<=' operator
                nodeStatement = Statement(nodeFeatures[nodeId], op='<=', val=nodeThresholds[nodeId])
                self._nodeStatements[nodeId] = nodeStatement


# ===================================
#       DATA GENERATION
# ===================================


def gen_data_rule(method: str = 'seneca', nSamples: int = 1000, nFeatures: int = 3, returnModel=False,
                  featureNames=None, randomState: int = 888):

    """ Generate synthetic binary classification data with ground truth decision rule explanations. The returned
    decision rule g.t. explanations are instances of the :code:`DecisionRule` class.

    :param str method: method to use for the generation of the ground truth explanations. Available:

        - 'seneca': g.t. explanations generated with the :class:`decisionRule.TransparentRuleClassifier` class. The
          method was presented in [Evaluating local explanation methods on ground truth, Riccardo Guidotti, 2021].

    :param nSamples: (int) number of samples to be generated.
    :param nFeatures: (int) total number of features in the generated data.
    :param returnModel: (bool) should the :code:`decisionRule.TransparentRuleClassifier` model used for the generation
        of the explanations be returned? Only used for the 'seneca' method.
    :param featureNames: (array-like) names of the generated features. If not provided, a list with the generated
        feature names will be returned by the function (necessary because the g.t. decision rules use them).
    :param randomState: (int) random state seed.
    :return:
        - X (ndarray) of shape (nSamples, nFeatures) Generated data.
        - y (ndarray) of shape (nSamples,) Binary data labels.
        - explanations (list) of :class:`decisionRule.DecisionRule` objects of length (nSamples). Generated ground
          truth explanations.
        - featureNames (list) list with the generated feature names (only if they were not specified).
        - model (:class:`decisionRule.TransparentRuleClassifier`) Model instance used to generate the data (returned if
          :code:`returnModel` is True only when :code:`method='seneca'`). """

    if method not in _AVAILABLE_DECISION_RULE_GEN_METHODS:
        raise ValueError(f'Method not available. Use one in {_AVAILABLE_DECISION_RULE_GEN_METHODS}')

    retFNames = False
    if featureNames is None:
        retFNames = True
        featureNames = _generate_feature_names(nFeatures)

    if method == 'seneca':
        data, targets, explanations, classifier = _gen_rule_dataset_seneca(nSamples=nSamples, nFeatures=nFeatures,
                                                                           randomState=randomState,
                                                                           featureNames=featureNames)

        if returnModel:
            if retFNames:
                return data, targets, explanations, featureNames, classifier
            else:
                return data, targets, explanations, classifier
        else:
            if retFNames:
                return data, targets, explanations, featureNames
            else:
                return data, targets, explanations


def load_data_rule(name):
    """ Loads (or downloads) and returns a real image dataset with available ground truth Decision Rule explanations.

    :param name: (str) dataset name. Available:
        -
    :return: A Dataset object. Read about it in the dataset module. """

    if name not in _AVAILABLE_DECISION_RULE_METRICS:
        raise ValueError(f'Dataset not available ({_AVAILABLE_DECISION_RULE_METRICS})')

    if name is None:
        # todo
        pass


def _gen_rule_dataset_seneca(nSamples=None, nFeatures=None, randomState=None, featureNames=None):
    """ g.t. explanations generated with the :class:`decisionRule.TransparentRuleClassifier` class. The
        method was presented in [Evaluating local explanation methods on ground truth, Riccardo Guidotti, 2021]. """

    # generate explanations with rules and binarize
    data, targets = make_classification(n_samples=nSamples, n_classes=2, n_features=nFeatures,
                                        n_informative=nFeatures, n_redundant=0, n_repeated=0,
                                        random_state=randomState)
    classifier = TransparentRuleClassifier(random_state=randomState)
    classifier.fit(data, targets, featureNames=featureNames)
    explanations = classifier.explain(data)

    return data, targets, explanations, classifier

# ===================================
#       EXPLANATION EVALUATION
# ===================================


def complete_rule_quality(gts: DecisionRule, rules: DecisionRule, eps: float = 0.1) -> float:
    """ Computes the complete rule quality (crq) between two decision rules. All 'Statements' in both rules must be
    binary (have upper and lower bounds). The metric is defined as the proportion of lower and upper bounds in a rule
    explanation whose that are eps-close to the respective lower and upper bounds (same feature) in the ground truth
    rule explanation amongst those that are != inf. Mathematically, given two rules :math:`e, \\tilde{e}` and a
    similarity threshold :math:`\\varepsilon`, the quality of :math:`e` with respect to :math:`\\tilde{e}` is:

    .. math::

        q(e, \\tilde{e}) = \\frac{1}{N_{\\not\\infty}} \\sum_{i=1}^{|e|}{\\delta_{\\varepsilon}(e_i, \\tilde e_i)},

    where

    .. math::

        \\delta_{\\varepsilon}(e_i, \\tilde{e}_i) =
                \\begin{cases}
                    1 & \\text{if } |e_i - \\tilde{e}_i| \\leq \\varepsilon \\wedge |e_i| \\neq \\infty \wedge
                        |\\tilde{e}_i| \\neq \\infty, \\newline
                    0 & \\text{otherwise}
                \\end{cases}

    Where :math:`N_{\\not \\infty}` is the number of lower and upper bounds that are different from :math:`\\infty` in
    both :math:`e` and :math:`\\tilde e`. More about this metric can be found in [Evaluating local explanation methods
    on ground truth, Riccardo Guidotti, 2021].

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
                if gt[feature].upperBound != np.inf and rule[feature].upperBound != np.inf:
                    nBounded += 1
                    if round(abs(gt[feature].upperBound - rule[feature].upperBound), 3) <= eps:
                        epsCloseBounds += 1
                # compare lower bounds
                if gt[feature].lowerBound != -np.inf and rule[feature].lowerBound != -np.inf:
                    nBounded += 1
                    if round(abs(gt[feature].lowerBound - rule[feature].lowerBound), 3) <= eps:
                        epsCloseBounds += 1
        res.append(epsCloseBounds / nBounded if nBounded != 0 else 0)

    return res[0] if len(res) == 1 else np.array(res).astype(np.float32)


def rule_scores(gts: DecisionRule, rules: DecisionRule, allFeatures, metrics=None, average=True, **kwargs) -> float:
    """ Quality metrics for :class:`decisionRule.DecisionRule` objects.

    :param gts: (DecisionRule or array-like of DecisionRules) ground truth decision rule/s.
    :param rules: (DecisionRule or array-like of DecisionRules) approximated decision rule/s.
    :param allFeatures: (array-like) names of all of the relevant features (i.e. :code:`create_rule_data.featureNames`)
    :param metrics: (array-like of str, default :code:`['fscore']`) metris to compute. Available:

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
    :return: (ndarray) specified metric/s in the original order. Can be of shape

        - (n_metrics,) if only one DecisionRule has been provided in both :code:`gts` and :code:`rules` or when both are
          array-like and :code:`average=True`.
        - (n_metrics, n_samples) if :code:`gts` and :code:`rules` are array-like and :code:`average=False`. """

    isArrayLike = isinstance(metrics, (list, np.ndarray, tuple))
    if not isArrayLike:
        if metrics not in _AVAILABLE_DECISION_RULE_METRICS:
            raise ValueError(f"'{metrics}' metric not valid. Use {_AVAILABLE_DECISION_RULE_METRICS}")
    else:
        for metric in metrics:
            if metric not in _AVAILABLE_DECISION_RULE_METRICS:
                raise ValueError(f"'{metric}' metric not valid. Use {_AVAILABLE_DECISION_RULE_METRICS}")

    crq, crqIndex = None, None
    if metrics is not None:
        if metrics == 'crq':
            return complete_rule_quality(gts, rules, **kwargs)
        elif isArrayLike and 'crq' in metrics:
            crq = complete_rule_quality(gts, rules, **kwargs)
            crqIndex = metrics.index('crq')
            del metrics[metrics.index('crq')]

    binaryGts = rule_to_feature_importance(gts, allFeatures)
    binaryRules = rule_to_feature_importance(rules, allFeatures)

    res = feature_importance_scores(binaryGts, binaryRules, metrics=metrics, average=average)

    if crq is not None:
        if isinstance(crq, np.ndarray):
            # multiple observations, insert as a column
            res = np.insert(res, crqIndex, crq, axis=1)
        elif isinstance(crq, float):
            # one observation
            res = np.insert(res, crqIndex, crq, axis=0)
    return res


# ===================================
#       UTILS
# ===================================

# todo parser to DecisionRule from other common representations

def rule_to_feature_importance(rules, allFeatures) -> list:
    """  Converts one or more :class:`decisionRule.DecisionRule` objects to feature importance vector/s. For each
    feature in *allFeatures*, the feature importance representation contains a 1 if there is a
    :class:'decisionRule.Statement' with that particular feature in the decision rule and 0 otherwise.

    :param rules: (:class:`decisionRule.DecisionRule` or (1, r) array-like of :class:`decisionRule.DecisionRule`) Rule/s
        to convert to feature importance vectors.
    :param allFeatures: (array-like of str) List with mg features (same as the rule features) whose order the returned
        array will follow. The features must match the ones used in the decision rules.
    :return: (binary ndarray of shape (n_features,) or shape (n_rules, n_features)). """

    if isinstance(rules, (list, ndarray, tuple)):
        res = []
        for rule in rules:
            res.append([1 if feature in rule else 0 for feature in allFeatures])
    elif isinstance(rules, DecisionRule):
        res = [1 if feature in rules else 0 for feature in allFeatures]
    else:
        raise ValueError('The rule is not a DecisionRule object nor array-like.')

    return np.array(res).astype(np.float32)


if __name__ == '__main__':

    from sklearn.datasets import make_classification

    # c = Statement('f', binary=True)
    # r = DecisionRule([c], result=Statement('a', op='=', val=5))

    ruleModel = TransparentRuleClassifier()
    X, y = make_classification(n_samples=50, n_classes=2, n_features=3, n_informative=3, n_redundant=0, n_repeated=0,
                               random_state=8)

    ruleModel.fit(X, y)
    print('MANUALLY GENERATING EXPLANATIONS WITH SENECA')
    print('Only one observation:')
    exp = ruleModel.explain(X[1].reshape(1, -1))
    print(exp[0])

    print('\n Multiple observations at once:')
    exp = ruleModel.explain(X[2:5])
    for e in exp:
        print(e)

    print('EXPLANATION EVALUATION')

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
