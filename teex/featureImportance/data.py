""" Module for synthetic and real datasets with available ground truth feature importance explanations. Also contains
methods and classes for decisionRule data manipulation.

All of the datasets must be instanced first. Then, when sliced, they all return the observations, labels and ground
truth explanations, respectively. """

from math import isnan

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler
from sympy import parse_expr, re, diff, Symbol

from teex._baseClasses._baseDatasets import _SyntheticDataset
from teex._baseClasses._baseClassifier import _BaseClassifier
from teex._utils._arrays import _scale_array
from teex._utils._misc import _generate_feature_names


class TransparentLinearClassifier(_BaseClassifier):
    """ Used on the higher level data generation class :class:`SenecaFI` (**use that and get it from there
    preferably**).

    Transparent, linear classifier with feature importances as explanations. This class also generates labeled
    data according to the generated random linear expression. Presented in [Evaluating local explanation methods on
    ground truth, Riccardo Guidotti, 2021]. """

    def __init__(self, randomState: int = 888):
        super().__init__()
        self.randomState = randomState
        # SymPy expression
        self.expression = None
        self.derivatives = None

        self.X = None
        self.y = None
        self.classIndices = None  # {class_0: [X_indices_class_0], class_1: [X_indices_class_1]}

        # Scalers for predicting probabilities
        self._scalerNeg = MinMaxScaler(feature_range=[0., 0.5])
        self._scalerPos = MinMaxScaler(feature_range=[0.5, 1.])

    def fit(self, nFeatures=None, featureNames=None, nSamples=100) -> None:
        """ Generates a random linear expression and random data labeled by the linear expression as a binary
        dataset.

        :param nFeatures: (int) number of features in the data.
        :param featureNames: (array-like) names of the features in the data.
        :param nSamples: (int) number of samples for the generated data.
        :return: (ndarray, ndarray) data of shape (n, m) and their respective labels of shape (n)
        """
        if featureNames is None and nFeatures is None:
            raise ValueError('The number of features or feature names should be provided.')
        elif featureNames is None:
            self.featureNames = _generate_feature_names(nFeatures)
        elif nFeatures is None:
            self.featureNames = featureNames
        elif len(featureNames) != nFeatures:
            raise ValueError("Provide all of the features' names.")

        self.expression = self._generate_expression()
        self.derivatives = self._differentiate_expression(self.expression)

        self.X, self.y = self._generate_data(nSamples=nSamples)

        self.classIndices = {dataClass: np.argwhere(self.y == dataClass).squeeze() for dataClass in np.unique(self.y)}

        # fit the scalers
        self._scalerNeg.fit(self.X[self.classIndices[0]].reshape(-1, 1))
        self._scalerPos.fit(self.X[self.classIndices[1]].reshape(-1, 1))

        return self.X, self.y

    def predict(self, data):
        """ Predicts label for observations. Class 1 if f(x) > 0 and 0 otherwise where x is a point to label and f()
        is the generated classification expression.

        :param data: (ndarray) observations to label, shape (k, m).
        :return: (ndarray) array of length n with binary labels.
        """
        return np.argmax(self.predict_proba(data), axis=1)

    def predict_proba(self, data):
        """ Get class probabilities by evaluating the expression f at 'data', normalizing the result and
        setting the probabilities as 1 - norm(f(data)), norm(f(data)).

        :param data: (ndarray) observations for which to obtain probabilities, shape (k, m).
        :return: (ndarray) array of shape (n, 2) with predicted class probabilities.
        """
        probs = []
        for point in data:
            value = self._evaluate_expression({f: v for f, v in zip(self.featureNames, point)})
            if isnan(value):
                value = 0
            else:
                if value > 0:
                    value = self._scalerPos.transform(np.array(value, dtype=np.float32).reshape(-1, 1))[0][0]
                else:
                    value = self._scalerNeg.transform(np.array(value, dtype=np.float32).reshape(-1, 1))[0][0]
            # bound all possible values
            value = max(min(value, 1.), 0.)
            probs.append([1 - value, value])
        return np.array(probs)

    def explain(self, data, newLabels=None):
        """ Get feature importance explanation as the gradient of the expression evaluated at the point (from the
        n 'training' observations) with the same class as 'obs' and closest to the decision boundary f = 0.

        The procedure is as follows: for each data observation x to explain, get the observation z from the 'training'
        data that is closer to the decision boundary and is of different class than x. Then, get the observation t from
        the 'training' data that is closer to z but of the same class as x. Finally, return the explanation for x as the
        gradient vector of f evaluated at t.

        :param data: (ndarray) array of k observations and m features, shape (k, m).
        :param newLabels: (ndarray, optional) precomputed data labels (binary ints) for 'data'. Shape (k).
        :return: (ndarray) (k, m) array of feature importance explanations. """

        if len(data.shape) != 2:
            raise ValueError('Observations to explain should have shape (k, m).')
        if newLabels is None:
            # compute labels
            newLabels = self.predict(data)

        distances = cdist(data, self.X, metric='euclidean')  # (k, n) where n is len(self.X)
        explanations = []
        for index, obs in enumerate(data):
            # get closest point of different class
            obsClass = newLabels[index]
            maskedDistances = distances[index].copy()
            maskedDistances[self.classIndices[obsClass]] = np.inf
            closestNot = np.argmin(maskedDistances)
            # get closest point to point of different class (same class as original data point)
            notObsClass = int(not newLabels[index])
            maskedDistances = cdist(self.X[closestNot].reshape(1, -1), self.X).squeeze()
            maskedDistances[self.classIndices[notObsClass]] = np.inf
            closest = np.argmin(maskedDistances)
            # evaluate gradient at 'closest'
            exp = self._evaluate_derivatives({f: v for f, v in zip(self.featureNames, self.X[closest])})
            explanations.append(exp)
        exps = np.array(explanations, dtype=np.float32)

        for i in range(len(self.featureNames)):
            # scale to (-1, 1) by feature max. and min. importance values
            exps[:, i] = np.round(np.interp(exps[:, i], (np.amin(exps[:, i]), np.amax(exps[:, i])), (-1, +1)), 4)

        return exps

    def _generate_expression(self):
        """ Generate a random linear expression following the procedure described in ["Evaluating local explanation
         methods on ground truth", Riccardo Guidotti 2020]. """

        unaryOps = ['{f}', '-{f}', '{f} ** 2', '{f} ** 3', 'sqrt({f})', 'log({f})', 'sign({f}', 'sin({f})', 'cos({f})',
                    'tan({f})', 'sinh({f})', 'cosh({f})', 'tanh({f})', 'asin({f})', 'acos({f})', 'atan({f})']
        binaryOps = ['{f1} + {f2}', '{f1} - {f2}', '{f1} * {f2}', '{f1} / {f2}', '{f1} ** {f2}']

        rng = np.random.default_rng(self.randomState)
        features = set(self.featureNames)
        expr = []
        for feature in features:
            if rng.uniform() < 0.5:
                expr.append(rng.choice(unaryOps).format(f=feature))
            else:
                # binary op
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
        return re(self.expression.evalf(subs=values))

    def _evaluate_derivatives(self, values: dict):
        """ Returns a list as the gradient vector of n features at a point 'values'. """
        grad = []
        for feature in values.keys():
            try:
                value = float(re(self.derivatives[feature].evalf(subs=values)))
            except TypeError or KeyError:
                # expression is not defined or feature does not play a role in the expression
                value = 0
            grad.append(value)
        return grad

    def _generate_data(self, nSamples):
        """ Generates two ndarrays of containing artificial data and its labels of shape nSamples * nFeatures and
        nFeatures, respectively. """
        rng = np.random.default_rng(self.randomState)
        data = np.array([rng.normal(scale=1, size=nSamples) for _ in range(len(self.featureNames))]).T
        labels = []
        for obs in data:
            labels.append(1 if self._evaluate_expression({f: v for f, v in zip(self.featureNames, obs)}) > 0 else 0)
        return data, np.array(labels, dtype=int)

    @staticmethod
    def _differentiate_expression(expression):
        """ Returns a dict with the first order _derivatives of a sympy expression w.r.t to each variable. """
        return {str(feature): diff(expression, feature) for feature in expression.atoms(Symbol)}


class SenecaFI(_SyntheticDataset):
    """ Generate synthetic binary classification tabular data with ground truth feature importance explanations. This
    method was presented in [Evaluating local explanation methods on ground truth, Riccardo Guidotti, 2021].

    From this class one can also obtain a trained transparent model (instance of :class:`TransparentLinearClassifier`).
    When sliced, this object will return

        - X (ndarray) of shape (nSamples, nFeatures) or (nFeatures). Generated data.
        - y (ndarray) of shape (nSamples,) or int. Generated binary data labels.
        - explanations (ndarray) of shape (nSamples, nFeatures) or (nFeatures). Generated g.t. feature importance
          explanations. For each explanation, the values are normalised to the [-1, 1] range.

    :param nSamples: (int) number of samples to be generated.
    :param nFeatures: (int) total number of features in the generated data.
    :param featureNames: (array-like) names of the generated features. If not provided, a list with the generated
        feature names will be returned by the function.
    :param randomState: (int) random state seed. """

    def __init__(self, nSamples: int = 200, nFeatures: int = 3, featureNames=None, randomState: int = 888) -> None:
        self.nSamples = nSamples
        self.nFeatures = nFeatures
        self.featureNames = _generate_feature_names(nFeatures) if featureNames is None else featureNames
        self.randomState = randomState

        self.X, self.y, self.exp, self.transparentModel = self._gen_seneca_dataset_fi()

    def __getitem__(self, item):
        if isinstance(item, (slice, int)):
            return self.X[item], self.y[item], self.exp[item]
        else:
            raise TypeError('Invalid argument type.')

    def __len__(self) -> int:
        return len(self.y)

    def _gen_seneca_dataset_fi(self):
        # explanations as gradient vectors around a decision boundary
        classifier = TransparentLinearClassifier(randomState=self.randomState)
        data, targets = classifier.fit(nSamples=self.nSamples, featureNames=self.featureNames)
        explanations = classifier.explain(data, newLabels=targets)

        return data, targets, explanations, classifier


def lime_to_feature_importance(exp, nFeatures, label=1):
    """ Convert from a ``lime.explanation.Explanation`` object to a np.array feature importance vector.

    :param lime.explanation.Explanation exp: explanation to convert to vector.
    :param label: (int, str) label of lime explanation. If lime explanations are generated by default, then it will be
        1.
    :param int nFeatures: number of features in the explanation
    :return: feature importance vector
    :rtype: np.ndarray """

    fiExp = np.zeros(nFeatures)
    for index, fi in exp.local_exp[label]:
        fiExp[index] = fi
    return fiExp


def scale_fi_bounds(x: np.ndarray):
    """ Map values of an 1D or 2D np.ndarray on certain conditions. The mapping is on a by-column basis. That is, each
    column will be separately scaled.::

        (for each column in ``x``)
        if values in the range [-1, 1] or [0, 1]       -> do nothing
        else:
            case 1: if values in the [0, inf] range    -> map to [0, 1]
            case 2: if values in the [-inf, 0] range   -> map to [-1, 1]
            case 3: if values in the [-inf, inf] range -> map to [-1, 1] """

    if len(x.shape) == 1 or (len(x.shape) == 2 and x.shape[0] == 1):
        # 1D
        return _scale_array(x)
    elif len(x.shape) == 2 and x.shape[0] != 1:
        # 2D
        totalNegVals = False
        for i in range(x.shape[1]):
            x[:, i], negVals = _scale_array(x[:, i])
            if negVals is True:
                totalNegVals = True
        return x, totalNegVals
    else:
        raise ValueError('Shape of array not supported.')
