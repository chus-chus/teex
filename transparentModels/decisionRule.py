import numpy as np
from sklearn.tree import DecisionTreeClassifier

from dataGen.datagen import gen_tabular_data
from transparentModels.baseClassifier import BaseClassifier

VALID_OPERATORS = {'=', '!=', '>', '<', '>=', '<='}


class Statement(object):
    """ An Statement follows the structure of 'feature' <operator> 'value'. It can also be binary as in
        feature <operatorLower> value1, feature <operatorHigher> value2 (if binary=True then 'value' and 'operator' are
        expected to be list-like with [lower, upper] and [operatorLower, operatorUpper], respectively. """

    def __init__(self, feature, operator, value, binary=False):
        self.feature = feature
        self.isBinary = binary
        if binary is True:
            self.lowerOperator = operator[0] if operator[0] in VALID_OPERATORS else None
            self.upperOperator = operator[1] if operator[1] in VALID_OPERATORS else None
            self.lowerValue = value[0]
            self.upperValue = value[1]
            if self.lowerValue is None or self.upperOperator is None:
                raise ValueError(f'Operator not valid. Choose from {VALID_OPERATORS}')
        else:
            if operator in VALID_OPERATORS:
                self.operator = operator
            else:
                raise ValueError(f'Operator not valid. Choose from {VALID_OPERATORS}')
            self.value = value

    def __eq__(self, other):
        if self.isBinary:
            return self.feature == other.feature and self.lowerOperator == other.lowerValue and \
                   self.upperOperator == other.upperValue and self.lowerValue == other.lowerValue and \
                   self.upperValue == other.upperValue
        else:
            return self.feature == other.feature and self.operator == other.operator and self.value == other.value

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        if self.isBinary:
            return f'{self.lowerValue} {self.lowerOperator} {self.feature} {self.upperOperator} {self.upperValue}'
        else:
            return f'{self.feature} {self.operator} {self.value}'

    def __hash__(self):
        return hash(str(self))


class DecisionRule(object):
    """ A conjunction of statements as conditionsn that imply a result. 'statements' must be a list-like array of
    Statement objects. Internally, the rule is represented as a dictionary of statements with the feature names as
    keys. A feature cannot have more than one Statement (Statements can be binary). 'result' is also an statement.
    """
    def __init__(self, statements=None, result=None):
        if statements is not None:
            self.statements = {}
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
        else:
            self.statements = None

        if result is not None and not isinstance(result, Statement):
            raise ValueError('Result must be an Statement.')

        self.result = result

    def __str__(self):
        return f"{', '.join([str(statement) for statement in self.statements])} -> {self.result}"

    def __len__(self):
        return len(self.statements)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return self.statements == other.statements and self.result == other.result

    def __ne__(self, other):
        return not self.__eq__(other)

    def add_condition(self, statement):
        """ Add Condition inplace to the conjunction """
        if statement.feature in self.statements:
            raise ValueError('Only one rule per feature is allowed.')
        else:
            self.statements[statement.feature] = statement

    def set_result(self, result):
        self.result = result


class RuleClassifier(BaseClassifier):
    """ Transparent, rule-based classifier with decision rules as explanations. """

    def __init__(self):
        super().__init__()
        self.model = DecisionTreeClassifier()

    def fit(self, data, target):
        self.model.fit(data, target)

    def predict(self, obs):
        return self.model.predict(obs)

    def predict_proba(self, obs):
        return self.model.predict_proba(obs)

    def predict_explain(self, obs):
        proba = self.predict_proba(obs)
        pass


if __name__ == '__main__':
    c = Statement('f', ['<=', '<'], [9, 13], binary=True)
    r = DecisionRule([c], result=Statement('a', '=', 5))
    r.show()

    ruleModel = RuleClassifier()
    X, y = gen_tabular_data()
    ruleModel.fit(X, y)

    ruleModel.predict(X[1])

