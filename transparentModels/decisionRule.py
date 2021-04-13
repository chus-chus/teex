import numpy as np

from dataGen.datagen import gen_tabular_data
from transparentModels.baseClassifier import BaseClassifier

VALID_OPERATORS = {'==', '!=', '>', '<', '>=', '<='}


class Condition(object):
    """ A condition follows the structure of if 'feature' <operator> 'value'. It can also be binary as in
        value1 <operatorLower> feature <operatorHigher> value2 (if binary=True then 'value' and 'operator' are expected
        to be list-like with [lower, upper] and [operatorLower, operatorUpper], respectively. """

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

    def show(self):
        if self.isBinary:
            print(f'{self.lowerValue} {self.lowerOperator} {self.feature} {self.upperOperator} {self.upperValue}')
        else:
            print(f'{self.feature} {self.operator} {self.value}')


class DecisionRule(object):
    """ A conjunction of conditions that imply a result. 'conditions' must be a list-like array of Condition objects.
        Internally, the rule is represented as a dictionary of conditions with the feature names as keys. A feature
        cannot have more than one condition (conditions can be binary).
    """
    def __init__(self, conditions=None, result=None):
        if conditions is not None:
            self.conditions = {}
            # if array-like, convert it into a dict
            if isinstance(conditions, (np.array, tuple, list)):
                for condition in conditions:
                    if not isinstance(condition, Condition):
                        raise ValueError('Conditions are not instances of the Condition class.')
                    elif condition.feature in self.conditions:
                        raise ValueError('Only one rule per feature is allowed.')
                    else:
                        self.conditions[condition.feature] = condition
            else:
                raise ValueError('Conditions not valid.')
        else:
            self.conditions = None
        self.result = result

    def add_condition(self, other):
        """ Add Condition inplace to the conjunction """
        if other.feature in self.conditions:
            raise ValueError('Only one rule per feature is allowed.')
        else:
            self.conditions[other.feature] = other

    def show(self):
        for condition in list(self.conditions.values()):
            condition.show()


class RuleClassifier(BaseClassifier):
    """ Transparent, rule-based classifier with decision rules as explanations. """

    def __init__(self):
        super().__init__()

    def fit(self, data, target):
        pass

    def predict(self, obs):
        pass

    def predict_proba(self, obs):
        pass

    def predict_explain(self, obs):
        pass


if __name__ == '__main__':
    ruleModel = RuleClassifier()
    X, y = gen_tabular_data()
    ruleModel.fit(X, y)

    ruleModel.predict(X[1])

