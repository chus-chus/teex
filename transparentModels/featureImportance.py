import numpy as np

from transparentModels.baseClassifier import BaseClassifier

from sympy import evalf
from sympy.parsing.sympy_parser import parse_expr


class LinearClassifier(BaseClassifier):
    """ Transparent, linear classifier with feature importances as explanations. """

    def __init__(self):
        super().__init__()
        # SymPy expression
        self.expression = None

    def fit(self, nFeatures=None, featureNames=None, randomState=888):
        """ Generates a random linear expression based on the provided number of features feature names. Note how the
        expression does not depend on any features, but rather the synthetic data should be later generated with
        the expression.

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

    def predict(self, obs):
        pass

    def predict_proba(self, obs):
        pass

    def explain(self, obs):
        pass

    def _generate_expression(self, randomState):
        """ Generate a random linear expressiion following the procedure described in ["Evaluating local explanation
         methods on ground truth", Riccardo Guidotti 2020. """

        unaryOps = ['{f}', '-{f}', 'sqrt({f})', 'log({f})', 'sign({f}', 'sin({f})', 'cos({f})', 'tan({f})',
                    'sinh({f})', 'cosh({f})', 'tanh({f})', 'asin({f})', 'acos({f})', 'atan({f})']
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


if __name__ == '__main__':
    m = LinearClassifier()
    featureNames = ['a', 'b', 'c']
    m.fit(featureNames=featureNames, randomState=1)

    print(m.expression)
    print(m._evaluate_expression({'a': 2, 'b': 3, 'c': -1}))
