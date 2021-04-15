import numpy as np
from sklearn.tree import DecisionTreeClassifier

from dataGen.datagen import gen_tabular_data
from transparentModels.baseClassifier import BaseClassifier

VALID_OPERATORS = {'=', '!=', '>', '<', '>=', '<='}


class Statement(object):
    f""" A Statement follows the structure of 'feature' <operator> 'value'. It can also be binary, like so: 
             value1 <operatorLower> feature <operatorHigher> value2 
        Valid operators are {VALID_OPERATORS}"""

    def __init__(self, feature, binary=False, lowOp='<', lowBound=-np.inf, upperOp='<=', upperBound=np.inf, op='=',
                 val=np.inf):
        """
        :param feature: (str) name of the feature for the Statement
        :param binary: (bool) is the statement binary?
        :param lowOp: (str) Operator for the lower bound (if binary)
        :param lowBound: Value of the upper bound (if binary)
        :param upperOp: (str) Operator for the upper bound (if binary)
        :param upperBound: Value of the lower bound (if binary)
        :param op: (str) Operator for the statement (if not binary)
        :param val: Value for the statement (if not binary)
        """
        self.feature = feature
        self.binary = binary
        if binary is True:
            self._check_operators([lowOp, upperOp])
            self.lowerOperator = lowOp
            self.upperOperator = upperOp
            self.lowerBound = lowBound
            self.upperBound = upperBound
        else:
            self._check_operators([op])
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
            return f'{self.lowerBound} {self.lowerOperator} {self.feature} {self.upperOperator} {self.upperBound}'
        else:
            return f'{self.feature} {self.operator} {self.value}'

    def __hash__(self):
        return hash(str(self))

    @staticmethod
    def _check_operators(operators):
        for op in operators:
            if op not in VALID_OPERATORS:
                raise ValueError(f"Operator '{op}' not valid. Choose from {VALID_OPERATORS}")


class DecisionRule(object):
    """ A conjunction of statements as conditions that imply a result. Internally, the rule is represented as a
    dictionary of statements with the feature names as keys. A feature cannot have more than one Statement (
    Statements can be binary). This class is capable of adapting previous Statements depending on new Statements that
    are added to it with the upsert method.
    """

    def __init__(self, statements=None, result=None):
        """
        :param statements: (list-like) Statement objects.
        :param result: Statement object.
        """
        self._statements = {}
        if statements is not None:
            # if array-like, convert it into a dict
            if isinstance(statements, (np.ndarray, tuple, list)):
                for statement in statements:
                    if not isinstance(statement, Statement):
                        raise ValueError('Statements are not instances of the Statement class.')
                    elif statement.feature in self._statements:
                        raise ValueError('Only one rule per feature is allowed.')
                    else:
                        self._statements[statement.feature] = statement
            else:
                raise ValueError('Statements not valid.')

        if result is not None and not isinstance(result, Statement):
            raise ValueError('Result must be an Statement.')

        self.result = result

    def __str__(self):
        return f"IF {', '.join([str(statement) for statement in self._statements.values()])} THEN {self.result}"

    def __len__(self):
        return len(self._statements)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return self._statements == other.statements and self.result == other.result

    def __ne__(self, other):
        return not self.__eq__(other)

    def __getitem__(self, feature):
        return self._statements[feature]

    def insert_condition(self, statement):
        """ Add Condition inplace to the conjunction.
        :param statement: Statement object
        """
        if statement.feature in self._statements:
            raise ValueError('A condition for this feature already exists.')
        else:
            self._statements[statement.feature] = statement

    def upsert_condition(self, statement, updateOperators=True):
        """ If a statement already exists within the rule, updates its bounds (replacing or defining them) and its
        operators if specified. If not, inserts the statement as a new condition. If an existing condition is of
        different type (binary / non-binary) as the new condition, the update fails. A bound update is only performed
        if the new bound/s != np.inf or -np.inf.

        :param statement: Statement object
        :param updateOperators: Should the operators be updated too? """

        if statement.feature not in self._statements:
            self._statements[statement.feature] = statement
        else:
            if statement.binary and self._statements[statement.feature].binary:
                if statement.upperBound != np.inf:
                    self._statements[statement.feature].upperBound = statement.upperBound
                if statement.lowerBound != -np.inf:
                    self._statements[statement.feature].lowerBound = statement.lowerBound
                if updateOperators:
                    self._statements[statement.feature].upperOperator = statement.upperOperator
                    self._statements[statement.feature].lowerOperator = statement.lowerOperator
            elif not statement.binary and not self._statements[statement.feature].binary:
                self._statements[statement.feature].value = statement.value
                if updateOperators:
                    self._statements[statement.feature].operator = statement.operator
            else:
                prevStatType = 'binary' if self._statements[statement.feature].binary else 'non-binary'
                newStatType = 'binary' if statement.binary else 'non-binary'
                raise ValueError(f'Cannot update {prevStatType} statement with new {newStatType} statement.')

    def set_result(self, result):
        self.result = result


class RuleClassifier(BaseClassifier):
    """ Transparent, rule-based classifier with decision rules as explanations. """

    def __init__(self):
        super().__init__()
        self.model = DecisionTreeClassifier()
        # dict. for each tree node, contains its learned condition as a "Statement" (None if node is a leaf)
        self.nodeStatements = None

    def fit(self, data, target, featureNames=None):
        """ Fits the classifier and automatically parses the learned tree structure into statements. """
        self.model.fit(data, target)

        if featureNames is None:
            self.featureNames = list(range(data.shape[1]))
        else:
            self.featureNames = featureNames

        self._parse_tree_structure()

    def predict(self, obs):
        """ Predicts the class for each observation.
        :param obs: array of n observations with m features and shape (n, m)
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
        :return: list with n DecisionRule objects
        """
        nodeIndicators = self.model.decision_path(obs)  # id's of the nodes for which each observation passes through
        rules = []
        # for each sample, retrieve its path and navigate it, looking at the precomputed decision splits
        for sampleId in range(len(obs)):
            nodePath = nodeIndicators.indices[nodeIndicators.indptr[sampleId]:nodeIndicators.indptr[sampleId+1]]
            rule = DecisionRule()
            for nodeId in nodePath:
                # node is not a leaf if None
                if self.nodeStatements[nodeId] is not None:
                    feature = self.nodeStatements[nodeId].feature
                    threshold = self.nodeStatements[nodeId].value
                    statement = Statement(self.featureNames[feature], lowOp='<', upperOp='<=', binary=True)
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
        """ Precomputes the learned tree splits and stores them as unary Statement objects. """
        nodeFeatures = self.model.tree_.feature
        nodeThresholds = self.model.tree_.threshold
        childrenRight = self.model.tree_.children_right
        childrenLeft = self.model.tree_.children_left

        self.nodeStatements = {nodeId: None for nodeId in range(self.model.tree_.node_count)}

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
                self.nodeStatements[nodeId] = nodeStatement


if __name__ == '__main__':
    c = Statement('f', binary=True)
    r = DecisionRule([c], result=Statement('a', op='=', val=5))

    ruleModel = RuleClassifier()
    X, y = gen_tabular_data(nFeatures=2)
    ruleModel.fit(X, y, featureNames=['alpha', 'beta'])

    exp = ruleModel.explain(X[1])
    print('Only one observation:')
    print(exp[0])
    print('\n Multiple observations at once:')
    exp = ruleModel.explain(X[2:5])

    for e in exp:
        print(e)
