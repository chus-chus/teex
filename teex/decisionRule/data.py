""" Module for synthetic and real datasets with available ground truth decision rule explanations. Also contains
methods and classes for decisionRule data manipulation.

All of the datasets must be instanced first. Then, when sliced, they all return the observations, labels and ground
truth explanations, respectively. """
import re
from typing import Tuple, List

import numpy as np
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier

from teex._baseClasses._baseDatasets import _SyntheticDataset
from teex._utils._misc import _generate_feature_names
from teex._baseClasses._baseClassifier import _BaseClassifier

_VALID_OPERATORS = {'=', '!=', '>', '<', '>=', '<='}
# operators for statements of shape: value1 <opLower> feature <opUpper> value2
_BINARY_OPERATORS = {'<', '<='}


class Statement(object):
    """ Class representing the atomic structure of a rule. A Statement follows the structure of 'feature'
    <operator> 'value'. It can also be binary, like so: ``value1 <lowOp> feature <upperOp> value2``. Valid 
    operators are {'=', '!=', '>', '<', '>=', '<='} or {'<', '<='} in the case of a binary statement. The class will
    store upper and lower bound values if the lower and upper operators are specified (both, just 1 is not valid). If
    the upper and lower operators are not specified, a unary Statement will be created.
    
    Although unary Statements (except '!=') have translation into single binary Statements, they are separately 
    represented for clarity. Moreover, unary Statements with operators '=' and '!=' are able to represent non-numeric
    values.
    
    :Example:
    
    >>> Statement('a',1.5)                                     # a = 1.5
    >>> Statement('a',1.5,op='!=')                             # a != 1.5
    >>> Statement('a',lowOp='<',lowB=2,upperOp='<',upperB=5)   # 2 < a < 5
    >>> Statement('a',lowOp='<',lowB=2)                        # 2 < a Wrong. Need to explicitly specify upper op
    >>> Statement('a',lowOp='<',lowB=2,upperOp='<')            # 2 < a < np.inf
    
    :param str feature: name of the feature for the Statement
    :param val: (float or str) Value for the statement (if not binary). Default ``np.inf``.
    :param str op: Operator for the statement (if not binary)
    :param str lowOp: Operator for the lower bound (if binary)
    :param float lowB: Value of the upper bound (if binary). Default ``-np.inf``.
    :param str upperOp: Operator for the upper bound (if binary)
    :param float upperB: Value of the lower bound (if binary). Default ``np.inf``.
    """

    def __init__(self, feature, val=np.inf, op='=', lowOp=None, lowB=-np.inf, upperOp=None, upperB=np.inf):
        self.feature = feature
        if lowOp is not None and upperOp is not None:
            self.binary = True
        elif (lowOp is None and upperOp is not None) or (lowOp is not None and upperOp is None):
            raise ValueError('Both operators must be indicated.')
        else:
            self.binary = False
        if self.binary:
            self._check_binary_operators(lowOp, upperOp)
            if lowB > upperB:
                raise ValueError('Lower bound cannot be greater than upper bound.')
            self.lowOp = lowOp
            self.upperOp = upperOp
            self.lowB = lowB
            self.upperB = upperB
        else:
            self._check_operator(op)
            self.op = op
            self.val = val

    def __eq__(self, other):
        if self.binary:
            return self.feature == other.feature and self.lowOp == other.lowB and self.upperOp == other.upperB and \
                   self.lowB == other.lowB and self.upperB == other.upperB
        else:
            return self.feature == other.feature and self.op == other.op and self.val == other.val

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        if self.binary:
            if self.lowB != -np.inf and self.upperB != np.inf:
                return f"{self.lowB} {self.lowOp} '{self.feature}' {self.upperOp} {self.upperB}"
            elif self.lowB != -np.inf:
                return f"{self.lowB} {self.lowOp} '{self.feature}'"
            elif self.upperB != np.inf:
                return f"'{self.feature}' {self.upperOp} {self.upperB}"
            else:
                return f"'{self.feature}' not bounded"
        else:
            if self.val != np.inf:
                return f"'{self.feature}' {self.op} {self.val}"
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
    dictionary of :class:`Statement` with the feature names as unique identifiers. A feature cannot have more than one
    :class:`Statement` (:class:`Statements` can be binary). This class is capable of adapting previous
    :class:`Statement` objects depending on new Statements that are added to it with the upsert method
    (see :func:`upsert_statement` method).

    :Example:

    >>> c1 = Statement('a',lowB=2,upperB=3)     # 2 < a < 3
    >>> r = DecisionRule([c1])
    >>> # update the bounds for the feature 'a'
    >>> c2 = Statement('a',lowB=3,upperB=5)
    >>> r.upsert_statement(c2,updateOperators=False)
    >>> # we can also insert new statements via upsert or insert
    >>> c3 = Statement('b',lowOp='<=',lowB=3,upperOp='<',upperB=6)
    >>> r.upsert_statement(c3)
    >>> # a Statement cannot be updated if one of them is different class as the other (binary / unary):
    >>> c4 = Statement('b', 3, op='>')
    >>> r.upsert_statement(c4) # THIS WILL RAISE AN ERROR!

    :param statements: (list-like of Statement objects) Statements as conditions that make the result be True.
    :param Statement result: Logical implication of the Decision Rule when all of the Statements are True.
    """

    def __init__(self, statements=None, result=None):
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
        """ Check if a Statement related to a feature is in the rule.

        :param str feature: Name of the Statement to check.
        """
        return feature in self.statements

    def delete_statement(self, feature) -> None:
        """ Deletes a Statement in the rule.

        :param str feature: name of the feature in the Statement to be deleted.
        """
        if feature in self.statements:
            del self.statements[feature]
        else:
            raise ValueError('A Statement with this feature name does not exist.')

    def get_features(self) -> list:
        """ Gets features in the Rule.

        :return list: feature names as identifiers of the Statements in the rule.
        """
        return list(self.statements.keys())

    def insert_statement(self, statement: Statement) -> None:
        """ Add Statement inplace to the conjunction.

        :param statement: Statement object
        """
        if statement.feature in self.statements:
            raise ValueError('A condition for this feature already exists.')
        else:
            self.statements[statement.feature] = statement

    def rename_statement(self, oldFeature, newFeature) -> None:
        """ Changes the identifier of a Statement.

        :param str oldFeature: id of the Statement to rename.
        :param str newFeature: new id of the Statement.
        """
        if newFeature in self.statements:
            raise ValueError(f'A Statement with the name {newFeature} already exists.')
        elif oldFeature in self.statements:
            self.statements[oldFeature] = newFeature
        else:
            raise ValueError(f'A Statement with the name {oldFeature} does not exist.')

    def replace_statement(self, oldFeature, newStatement: Statement) -> None:
        """ Replaces a Statement with another.

        :param str oldFeature: identifier of the Statement to replace.
        :param Statement newStatement: new statement.
        """

        if oldFeature not in self.statements:
            raise ValueError(f'A condition for the feature {oldFeature} does not exist. Use the insert method instead.')
        else:
            self.statements[oldFeature] = newStatement

    def set_result(self, result) -> None:
        """ Sets the result for the Decision Rule.

        :param Statement result: statement as logical implication.
        """
        if not isinstance(result, Statement):
            raise TypeError('results should be a Statement object.')
        else:
            self.result = result

    def upsert_statement(self, statement: Statement, updateOperators: bool = True) -> None:
        """ If a statement already exists within the rule, updates its bounds (replacing or defining them) and its
        operators if specified. If not, inserts the statement as a new condition. If an existing condition is of
        different type (binary / non-binary) as the new condition, the update fails. A bound update is only performed
        if the new bound/s != np.inf or -np.inf.

        :param statement: Statement object to upsert
        :param updateOperators: Should the operators be updated too? """

        if statement.feature not in self.statements:
            self.statements[statement.feature] = statement
        else:
            if statement.binary and self.statements[statement.feature].binary:
                if statement.upperB != np.inf:
                    self.statements[statement.feature].upperB = statement.upperB
                if statement.lowB != -np.inf:
                    self.statements[statement.feature].lowB = statement.lowB
                if updateOperators:
                    self.statements[statement.feature].upperOp = statement.upperOp
                    self.statements[statement.feature].lowOp = statement.lowOp
            elif not statement.binary and not self.statements[statement.feature].binary:
                self.statements[statement.feature].val = statement.val
                if updateOperators:
                    self.statements[statement.feature].op = statement.op
            else:
                prevStatType = 'binary' if self.statements[statement.feature].binary else 'non-binary'
                newStatType = 'binary' if statement.binary else 'non-binary'
                raise ValueError(f'Cannot update {prevStatType} statement with new {newStatType} statement.')


class TransparentRuleClassifier(_BaseClassifier):
    """ Used on the higher level data generation class :class:`teex.featureImportance.data.SenecaFI`
    (**use that and get it from there preferably**).

    Transparent, rule-based classifier with decision rules as explanations. For each prediction, the associated
    ground truth explanation is available with the :func:`explain` method. Follows the sklearn API. Presented in
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
        :param featureNames: (array-like) names of the features in the data. If not specified, they will be created.
            Stored in ``self.featureNames``.
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
        :return np.ndarray: array of n predicted labels
        """
        return self.model.predict(obs)

    def predict_proba(self, obs):
        """ Predicts probability that each observation belongs to each of the c classes.

        :param obs: array of n observations with m features and shape (n, m)
        :return np.ndarray: array of n probability tuples of length c
        """
        return self.model.predict_proba(obs)

    def explain(self, obs):
        """  Explain observations' predictions with decision rules.

        :param obs: array of n observations with m features and shape (n, m)
        :return: list with n :class:`DecisionRule` objects """
        nodeIndicators = self.model.decision_path(obs)  # id's of the nodes for which each observation passes through
        rules = []
        # for each sample, retrieve its path and navigate it, looking at the precomputed decision splits
        for sampleId in range(len(obs)):
            nodePath = nodeIndicators.indices[nodeIndicators.indptr[sampleId]:nodeIndicators.indptr[sampleId + 1]]
            rule = DecisionRule()
            for nodeId in nodePath:
                # node is not a leaf if None
                if self._nodeStatements[nodeId] is not None:
                    feature = self._nodeStatements[nodeId].feature
                    threshold = self._nodeStatements[nodeId].val
                    statement = Statement(self.featureNames[feature], lowOp='<', upperOp='<=')
                    # define bounds. Remember that the tree splits are of the form: feature '<=' val
                    if obs[sampleId][feature] <= threshold:
                        statement.upperB = round(threshold, 3)
                    else:
                        statement.lowB = round(threshold, 3)
                    # we create binary statements and will update the bounds as we traverse the tree's decision
                    rule.upsert_statement(statement, updateOperators=False)
            rule.set_result(Statement('Class', val=self.predict(obs[sampleId].reshape(1, -1))[0], op='='))
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
                # all split nodes check with the '<=' op
                nodeStatement = Statement(nodeFeatures[nodeId], val=nodeThresholds[nodeId], op='<=')
                self._nodeStatements[nodeId] = nodeStatement


class SenecaDR(_SyntheticDataset):
    """ Generate synthetic binary classification data with ground truth decision rule explanations. The returned
    decision rule g.t. explanations are instances of the :class:`DecisionRule` class.

    Ground truth explanations are generated with the :class:`TransparentRuleClassifier` class. The method was presented
    in [Evaluating local explanation methods on ground truth, Riccardo Guidotti, 2021]. From this class one can also
    obtain a trained transparent model (instance of :class:`TransparentRuleClassifier`).

    When sliced, this object will return
        - X (ndarray) of shape (nSamples, nFeatures) or (nFeatures). Generated data.
        - y (ndarray) of shape (nSamples,) or int. Binary data labels.
        - explanations (list) of :class:`DecisionRule` objects of length (nSamples) or :class:`DecisionRule` object.
          Generated ground truth explanations.

    :param int nSamples: number of samples to be generated.
    :param int nFeatures: total number of features in the generated data.
    :param featureNames: (array-like) names of the generated features. If not provided, a list with the generated
        feature names will be returned by the function (necessary because the g.t. decision rules use them).
    :param int randomState: random state seed. """

    def __init__(self, nSamples: int = 1000, nFeatures: int = 3, featureNames=None, randomState: int = 888) -> None:
        self.nSamples = nSamples
        self.nFeatures = nFeatures
        self.featureNames = _generate_feature_names(nFeatures) if featureNames is None else featureNames
        self.randomState = randomState

        self.X, self.y, self.exp, self.transparentModel = self._gen_dataset_seneca_dr()

    def __getitem__(self, item):
        if isinstance(item, (slice, int)):
            return self.X[item], self.y[item], self.exp[item]
        else:
            raise TypeError('Invalid argument type.')

    def __len__(self) -> int:
        return len(self.y)

    def _gen_dataset_seneca_dr(self):
        """ g.t. explanations generated with the :class:`TransparentRuleClassifier` class. The
            method was presented in [Evaluating local explanation methods on ground truth, Riccardo Guidotti, 2021]. """

        # generate explanations with rules and binarize
        data, targets = make_classification(n_samples=self.nSamples, n_classes=2, n_features=self.nFeatures,
                                            n_informative=self.nFeatures, n_redundant=0, n_repeated=0,
                                            random_state=self.randomState)
        classifier = TransparentRuleClassifier(random_state=self.randomState)
        classifier.fit(data, targets, featureNames=self.featureNames)
        explanations = classifier.explain(data)

        return data, targets, explanations, classifier


# Utils for data manipulation:

def rule_to_feature_importance(rules, allFeatures) -> np.ndarray:
    """  Converts one or more :class:`DecisionRule` objects to feature importance vector/s. For each
    feature in *allFeatures*, the feature importance representation contains a 1 if there is a
    :class:'Statement' with that particular feature in the decision rule and 0 otherwise.

    :param rules: (:class:`DecisionRule` or (1, r) array-like of :class:`DecisionRule`) Rule/s
        to convert to feature importance vectors.
    :param allFeatures: (array-like of str) List with m features (same as the rule features) whose order the returned
        array will follow. The features must match the ones used in the decision rules.
    :return: (binary ndarray of shape (n_features,) or shape (n_rules, n_features)). """

    if isinstance(rules, (list, np.ndarray, tuple)):
        res = []
        for rule in rules:
            res.append([1 if feature in rule else 0 for feature in allFeatures])
    elif isinstance(rules, DecisionRule):
        res = [1 if feature in rules else 0 for feature in allFeatures]
    else:
        raise ValueError('The rule is not a DecisionRule object nor array-like.')

    return np.array(res).astype(np.float32)


def _induce_binary_statement(feature, operator, value):
    if operator == '<' or operator == '<=':
        s = Statement(feature, lowOp='<', upperOp=operator, upperB=float(value))
    elif operator == '>' or operator == '>=':
        op = '<' if operator == '>' else '<='
        s = Statement(feature, upperOp='<', lowOp=op, lowB=float(value))
    else:
        raise ValueError('Operator for binary statement not valid.')
    return s


def _generate_binary_statement(feature, op1, val1, op2, val2):
    """ Generates binary statement from operators, a feature and values. """

    if op1 == '>' or op1 == '>=':
        lowOp = '<' if op1 == '>' else '<='
        s = Statement(feature, lowOp=lowOp, lowB=float(val1), upperOp=op2, upperB=float(val2))
    elif op2 == '>' or op2 == '>=':
        lowOp = '<' if op2 == '>' else '<='
        s = Statement(feature, lowOp=lowOp, lowB=float(val2), upperOp=op1, upperB=float(val1))
    else:
        raise ValueError('Operator for binary statement not valid.')
    return s


def clean_binary_statement(bounds: list):
    """ Parses binary statement edge cases from a list of operators and values. Checks if the edge cases
    occur for any pair of operator and value. Does not fix errors with bounds != or =.

    f > 3 & f > 4 TRANSFORMS INTO f > 4
    f > 3 & f >= 4 TRANSFORMS INTO f >= 4

    f < 3 & f < 4 TRANSFORMS INTO f < 3
    f <= 3 & f < 4 TRANSFORMS INTO f <= 3

    :param list bounds: list with bounds i.e. [(op, val), ..., (op, val)]
    :return: op1, val1, op2, val2
    """

    op1, val1 = '>', -np.inf
    op2, val2 = '<', np.inf
    for op, val in bounds:
        if op == '>=' or op == '>':
            if val > val1:
                op1, val1 = op, val
        elif op == '<=' or op == '<':
            if val < val2:
                op2, val2 = op, val
        else:
            raise ValueError('Invalid bound operator for binary statement.')

    return op1, val1, op2, val2


def _get_statements_dict(strRule, statementType='binary') -> dict:
    """ Returns a dictionary with Statements from a string. """

    rules = re.split(r'&', strRule) if re.search(r'&', strRule) else [strRule]
    statements = {}

    # parse rules
    for rule in rules:
        # remove all whitespaces
        rule = re.sub(r'\s', '', rule)
        # match the feature in a group and the operator in another one
        matches = re.search(r'([\w]*)(<=|!=|>=|<|>|=)', rule)
        feature = matches.group(1)
        op = matches.group(2)
        value = rule[matches.span()[1]:]
        try:
            value = float(value)
        except ValueError:
            pass
        if statementType == 'unary':
            statements[feature] = Statement(feature, val=value, op=op)
        elif statementType == 'binary':
            if feature in statements:
                statements[feature].append((op, value))
            else:
                statements[feature] = [(op, value)]

    # generate binary statements with collected information
    if statementType == 'binary':
        for feature, conditions in statements.items():
            if len(conditions) == 1:
                statements[feature] = _induce_binary_statement(feature, conditions[0][0], conditions[0][1])
            elif len(conditions) == 2:
                op1, val1, op2, val2 = clean_binary_statement(conditions)  # remove edge cases
                statements[feature] = _generate_binary_statement(feature, op1, val1, op2, val2)
            else:
                try:
                    op1, val1, op2, val2 = clean_binary_statement(conditions)
                    statements[feature] = _generate_binary_statement(feature, op1, val1, op2, val2)
                except ValueError:
                    raise ValueError('Too many statements for one feature.')

    return statements


def str_to_decision_rule(strRule: str, ruleType: str = 'binary') -> DecisionRule:
    """ Converts a string representing a rule into a DecisionRule object. The string must contain the individual feature
    bounds separated by '&'. For each feature bound, the feature must appear first. If ``ruleType='binary'``,
    it is not necessary to explicitly specify both bounds: the missing one will be induced. To imply a result, use '->'
    and follow it with a statement representation. This method is robust to situations like
    ``feature > 3 & feature > 4`` and missing whitespaces.

    :Example:

    >>> r = 'a != 2.5 -> res > 3'
    >>> print(str_to_decision_rule(r,'unary'))
    >>> r = 'a <= 2.5 & a > 1 -> res > 3'
    >>> print(str_to_decision_rule(r,'binary'))
    >>> r = 'a <= 2.5 & a > 1 & b > 1 -> res > 3 & res <= 5'
    >>> print(str_to_decision_rule(r,'binary'))
    >>> r = 'a <= 2.5 & a > 1 & b > 1 -> res = class0'
    >>> print(str_to_decision_rule(r,'binary'))
    >>> print(str_to_decision_rule('d > 1 & d > 3 & d >= 4 & c < 4 & c < 3 & c <= 2-> home > 1 & home < 3')) # is robust

    :param str strRule: string to convert to rule.
    :param str ruleType: type of the Statement objects contained within the generated DecisionRule object. """

    if ruleType != 'binary' and ruleType != 'unary':
        raise ValueError(f"ruleType not valid. Use {['binary', 'unary']}")

    # check if there is a result and parse accordingly
    res = re.search(r'->', strRule)
    if res:
        resultStr = strRule[res.span()[1]:]
        resultsType = 'binary' if re.search(r'&', resultStr) else 'unary'
        resultStatement = list(_get_statements_dict(resultStr, resultsType).values())[0]
        strRule = strRule[:res.span()[0]]
    else:
        resultStatement = None

    statements = _get_statements_dict(strRule, ruleType)

    return DecisionRule([statement for statement in statements.values()], resultStatement)


def rulefit_to_decision_rule(rules, minImportance: float = 0., minSupport: float = 0.) -> Tuple[
        List[DecisionRule], list]:
    """ Transforms rules computed with the RuleFit algorithm (only from
    `this <https://github.com/christophM/rulefit>`_ implementation) into DecisionRule objects.

    :Example:

    >>> import pandas as pd
    >>> from rulefit import RuleFit
    >>>
    >>> boston_data = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv')
    >>> y = boston_data.medv.values
    >>> features = boston_data.columns
    >>> X = boston_data.drop("medv", axis=1).values
    >>>
    >>> rf = RuleFit()
    >>> rf.fit(X, y, feature_names=features)
    >>>
    >>> dRules, _ = rulefit_to_decision_rule(rf.get_rules(), rf.predict(X))

    :param pd.DataFrame rules: rules computed with the .get_rules() method of RuleFit. Default 0.
    :param float minImportance: minimum importance for a rule to have to be transformed. Default 0.
    :param float minSupport: minimum support for a rule to have to be transformed.
    :return:
        - (list) parsed DecisionRules
        - (list) indexes of skipped rows (because of exceptions such as 'home < 1 & home > 3'). """

    skippedRows = []
    decisionRules = []
    nLinears = 0
    for index, rule in rules.iterrows():
        if rule['type'] == 'rule':
            if rule['importance'] >= minImportance and rule['support'] >= minSupport:
                try:
                    rule = str_to_decision_rule(rule['rule'])
                    decisionRules.append(rule)
                except ValueError:
                    skippedRows.append(index)
        else:
            nLinears += 1
    return decisionRules, skippedRows
