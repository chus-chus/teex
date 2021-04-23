import numpy as np

from transparentModels.baseClassifier import BaseClassifier

from sympy import evalf, diff, Symbol
from sympy.parsing.sympy_parser import parse_expr


class LinearClassifier(BaseClassifier):
    """ Transparent, linear classifier with feature importances as explanations. """

    def __init__(self):
        super().__init__()
        # SymPy expression
        self.expression = None

        self.derivatives = None

    def fit(self, data, nFeatures=None, featureNames=None, randomState=888):
        """ Generates a random linear expression based on the provided number of features feature names.
        # todo fitting returns data labels

        :param data: (ndarray) data for which to generate labels based on the generated expression.
        :param nFeatures: (int) number of features in the data.
        :param featureNames: (array-like) names of the features in the data.
        :param randomState: (optional, int) random state for the generation of the linear expression.
        """
        if featureNames is None and nFeatures is None:
            raise ValueError('The number of features or feature names should be provided.')
        elif featureNames is None:
            self.featureNames = [str(i) for i in range(nFeatures)]
        elif nFeatures is None:
            self.featureNames = featureNames
        elif len(featureNames) != nFeatures:
            raise ValueError("Provide all of the features' names.")

        self.expression = self._generate_expression(randomState)
        self.derivatives = self._differentiate_expression(self.expression)

    def predict(self, obs):
        pass

    def predict_proba(self, obs):
        pass

    def explain(self, obs):
        pass

    def _generate_expression(self, randomState):
        """ Generate a random linear expressiion following the procedure described in ["Evaluating local explanation
         methods on ground truth", Riccardo Guidotti 2020. """

        unaryOps = ['{f}', '-{f}', '{f} ** 2', '{f} ** 3', 'sqrt({f})', 'log({f})', 'sign({f}', 'sin({f})', 'cos({f})',
                    'tan({f})', 'sinh({f})', 'cosh({f})', 'tanh({f})', 'asin({f})', 'acos({f})', 'atan({f})']
        binaryOps = ['{f1} + {f2}', '{f1} - {f2}', '{f1} * {f2}', '{f1} / {f2}', '{f1} ** {f2}']

        rng = np.random.default_rng(randomState)
        features = set(self.featureNames)
        expr = []
        for feature in features:
            if rng.uniform() < 0.5:
                expr.append(rng.choice(unaryOps).format(f=feature))
            else:
                # binary operator
                op = rng.choice(binaryOps)
                # choose second feature
                feature2 = rng.choice(list(features - set(feature)))
                # decide order of set
                if rng.uniform() < 0.5:
                    expr.append(op.format(f1=feature, f2=feature2))
                else:
                    expr.append(op.format(f1=feature2, f2=feature))

        return parse_expr('+'.join(expr))

    def _evaluate_expression(self, values: dict):
        return self.expression.evalf(subs=values)

    @staticmethod
    def _differentiate_expression(expression):
        """ Returns a dict with the first order derivatives of a sympy expression w.r.t to each variable. """
        return {feature: diff(expression, feature) for feature in expression.atoms(Symbol)}


if __name__ == '__main__':
    m = LinearClassifier()
    features = ['a', 'b', 'c', 'd']
    m.fit(None, featureNames=features, randomState=1)

    print(m.expression)
    print(m.derivatives)
    print(m._evaluate_expression({'a': 2, 'b': 3, 'c': -1}))
