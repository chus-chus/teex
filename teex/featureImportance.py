""" The feature importance module. Contains all of the methods regarding this explanation type: synthetic data
generation, evaluation of explanations and related utils. """

import warnings

import numpy as np
import sklearn.metrics as met

from sympy import diff, Symbol, re
from sympy.parsing.sympy_parser import parse_expr
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler
from math import isnan

# noinspection PyProtectedMember
from teex._baseClasses._baseClassifier import _BaseClassifier
# noinspection PyProtectedMember
from teex._utils._misc import _generate_feature_names
# noinspection PyProtectedMember
from teex._utils._arrays import _binarize_arrays
from teex.datasets import Newsgroup

_AVAILABLE_FEATURE_IMPORTANCE_METRICS = {'fscore', 'prec', 'rec', 'cs', 'auc'}
_AVAILABLE_FI_GEN_METHODS = {'seneca'}
_AVAILABLE_FEATURE_IMPORTANCE_DATASETS = {'newsgroup'}

# ===================================
#       TRANSPARENT MODEL
# ===================================


class TransparentLinearClassifier(_BaseClassifier):
    """ Transparent, linear classifier with feature importances as explanations. This class also generates labeled
    data according to the generated random linear expression. Presented in [Evaluating local explanation methods on
    ground truth, Riccardo Guidotti, 2021]. """

    def __init__(self, randomState=888):
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

    def fit(self, nFeatures=None, featureNames=None, nSamples=100):
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
        return np.round(np.interp(exps, (np.amin(exps), np.amax(exps)), (-1, +1)), 4)

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
        return re(self.expression.evalf(subs=values))

    def _evaluate_derivatives(self, values: dict):
        """ Returns a list as the gradient vector of n features at a point 'values'. """
        grad = []
        for feature in values.keys():
            try:
                value = float(re(self.derivatives[feature].evalf(subs=values)))
            except TypeError:
                # expression is not defined
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


# ===================================
#       DATA GENERATION
# ===================================

def gen_data_fi(method: str = 'seneca', nSamples: int = 200, nFeatures: int = 3, returnModel=False, featureNames=None,
                randomState: int = 888):
    """ Generate synthetic binary classification tabular data with ground truth feature importance explanations.

    :param method: (str) method to use for the generation of the ground truth explanations. Available:

        - 'seneca': g.t. explanations generated with the :class:`featureImportance.TransparentLinearClassifier` class.
          The method was presented in [Evaluating local explanation methods on ground truth, Riccardo Guidotti, 2021].

    :param nSamples: (int) number of samples to be generated.
    :param nFeatures: (int) total number of features in the generated data.
    :param returnModel: (bool) should the :code:`featureImportance.TransparentLinearClassifier` model used for the
        generation of the explanations be returned? Only used for the 'seneca' method.
    :param featureNames: (array-like) names of the generated features. If not provided, a list with the generated
        feature names will be returned by the function.
    :param randomState: (int) random state seed.
    :return:
        - X (ndarray) of shape (nSamples, nFeatures) Generated data.
        - y (ndarray) of shape (nSamples,) Generated binary data labels.
        - explanations (ndarray) of shape (nSamples, nFeatures) Generated g.t. feature importance explanations. For each
          explanation, the values are normalised to the [-1, 1] range.
        - featureNames (list) list with the generated feature names (only if they were not specified).
        - model (:class:`featureImportance.TransparentLinearClassifier`) Model instance used to generate the data
          (returned if :code:`returnModel` is True only when :code:`method='seneca'`). """

    if method not in _AVAILABLE_FI_GEN_METHODS:
        raise ValueError(f'Method not available. Use one in {_AVAILABLE_FI_GEN_METHODS}')

    retFNames = False
    if featureNames is None:
        retFNames = True
        featureNames = _generate_feature_names(nFeatures)

    if method == 'seneca':
        data, targets, explanations, classifier = _gen_seneca_dataset_fi(nSamples=nSamples, featureNames=featureNames,
                                                                         randomState=randomState)
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


def load_data_fi(name: str = 'newsgroup'):
    """ Loads (or downloads) and returns a dataset with available ground truth feature importance explanations.

    :param name: (str) dataset name. Available:

        - 'newsgroup': Binary classification dataset from [] with g.t. explanations as word importances.
          # todo further explan.

    :return: A Dataset object. Read about it in the dataset module. """

    if name not in _AVAILABLE_FEATURE_IMPORTANCE_DATASETS:
        raise ValueError(f'Dataset not available ({_AVAILABLE_FEATURE_IMPORTANCE_DATASETS})')

    if name == 'newsgroup':
        return Newsgroup()


def _gen_seneca_dataset_fi(nSamples=None, featureNames=None, randomState=None):
    """ g.t. explanations generated with the :class:`featureImportance.TransparentLinearClassifier` class.
        The method was presented in [Evaluating local explanation methods on ground truth, Riccardo Guidotti, 2021]. """

    # explanations as gradient vectors around a decision boundary
    classifier = TransparentLinearClassifier(randomState=randomState)
    data, targets = classifier.fit(nSamples=nSamples, featureNames=featureNames)
    explanations = classifier.explain(data, newLabels=targets)

    return data, targets, explanations, classifier


# ===================================
#       EXPLANATION EVALUATION
# ===================================


def _individual_fi_metrics(gt, pred, binGt, binPred, metric, predsNegative, thresholdType):
    """ Classification and real vector metrics. If metric='auc' and predsNegative=True, 'pred' is modified accordingly
    (see documentation of featureImportance.feature_importance_scores).

     :param gt: (ndarray) of shape (nFeatures,). Ground truth (real or binary) vector.
     :param pred: (ndarray) of shape (nFeatures,). Predicted (real or binary) vector.
     :param binGt: (ndarray) of shape (nFeatures,). Ground truth (binary) vector.
     :param binPred: (ndarray) of shape (nFeatures,). Predicted (binary) vector.
     :param metric: (str) in ['fscore', 'prec', 'rec', 'cs', 'auc'] metric to compute.
     :param predsNegative: (bool) whether 'pred' contains negative values or not.
     :param thresholdType: (str) in ['abs', 'thres].
     :return: (float) selected metric. """

    if metric == 'fscore':
        return met.fbeta_score(binGt, binPred, beta=1)
    elif metric == 'prec':
        return met.precision_score(binGt, binPred)
    elif metric == 'rec':
        return met.recall_score(binGt, binPred)
    elif metric == 'cs':
        return cosine_similarity(gt, pred)
    elif metric == 'auc':
        if predsNegative is True:
            pred = np.abs(pred) if thresholdType == 'abs' else np.where(pred < 0, 0, pred)
        return met.roc_auc_score(binGt, pred)
    

def _check_correct_array_values(binGt, binPred, gt, pred):
    """ Checks whether the passed arrays contain all 0s. """

    emptyBGt, emptyBPred, emptyGt, emptyPred = False, False, False, False
    if (binGt == 0).all():
        emptyBGt = True
    if (binPred == 0).all():
        emptyBPred = True
    if emptyBGt and (gt == 0).all():
        emptyGt = True
    if emptyBPred and (pred == 0).all():
        emptyPred = True

    return emptyBGt, emptyBPred, emptyGt, emptyPred


def feature_importance_scores(gts, preds, metrics=None, average=True, thresholdType='abs', binThreshold=0.5):
    """ Computes quality metrics between one or more feature importance vectors. The values in the vectors must be
    bounded in [0, 1] or [-1, 1] (to indicate negative importances in the second case).

    For the computation of the precision, recall and FScore, the vectors are binarized to simulate a classification
    setting depending on the param. :code:`thresholdType`. In the case of ROC AUC, the ground truth feature importance
    vector will be binarized as in the case of 'precision', 'recall' and 'FScore' and the predicted feature importance
    vector entries will be considered as prediction scores. If the predicted vectors contain negative values, these will
    be either mapped to 0 or taken their absolute value (depending on the chosen option in the param.
    :code:`thresholdType`).

    :param np.ndarray gts: (1d np.ndarray or 2d np.ndarray of shape (n_features, n_samples)) ground truth feature
                            importance vectors.
    :param np.ndarray preds: (1d np.ndarray or 2d np.ndarray of shape (n_features, n_samples)) predicted feature
                              importance vectors.
    :param metrics:
        (str or array-like of str) metric/s to be computed. Available metrics are

            - 'fscore': Computes the F1 Score between the ground truths and the predicted vectors.
            - 'prec': Computes the Precision Score between the ground truths and the predicted vectors.
            - 'rec': Computes the Recall Score between the ground truths and the predicted vectors.
            - 'auc': Computes the ROC AUC Score between the ground truths and the predicted vectors.
            - 'cs': Computes the Cosine Similarity between the ground truths and the predicted vectors.

        The vectors are automatically binarized for computing recall, precision and fscore.
    :param bool average: (default :code:`True`) Used only if :code:`gt` and :code:`rule` contain multiple
                    observations. Should the computed metrics be averaged across all the samples?
    :param str thresholdType:
        Options for the binarization of the features for the computation of 'fscore', 'prec', 'rec' and 'auc'.

            - 'abs': features with absolute value <= :code:`binThreshold` will be set to 0 and 1 otherwise. For the
              predicted feature importances in the case of 'auc', their absolute value will be taken.
            - 'thres': features <= :code:`binThreshold` will be set to 0, 1 otherwise. For the `predicted` feature
              importances in the case of 'auc', negative values will be cast to 0 and the others left *as-is*.

    :param float binThreshold:
        (in [-1, 1]) Threshold for the binarization of the features for the computation of 'fscore', 'prec', 'rec' and
        'auc'. The binarization depends on both this parameter and :code:`thresholdType`.
        If :code:`thresholdType = 'abs'`, the value cannot be negative.
    :return: (ndarray of shape (n_metrics,) or (n_samples, n_metrics)) specified metric/s in the indicated order. """

    # todo document how the edges cases are dealt with
    
    if metrics is None:
        metrics = ['fscore']
    elif isinstance(metrics, str):
        metrics = [metrics]

    for metric in metrics:
        if metric not in _AVAILABLE_FEATURE_IMPORTANCE_METRICS:
            raise ValueError(f"'{metric}' metric not valid. Use {_AVAILABLE_FEATURE_IMPORTANCE_METRICS}")

    if not isinstance(gts, np.ndarray) or not isinstance(preds, np.ndarray):
        raise TypeError('Ground truths and predictions must be np.ndarrays.')

    if (gts > 1).any() or (gts < -1).any():
        raise ValueError('Some g.t. feature importance vectors are not bounded in [-1, 1].')

    if (preds > 1).any() or (preds < -1).any():
        raise ValueError('Some predicted feature importance vectors are not bounded in [-1, 1].')
    elif (gts < 0).any():
        predsNegative = True
    else:
        predsNegative = False

    binaryGts = _binarize_arrays(gts, method=thresholdType, threshold=binThreshold)
    binaryPreds = _binarize_arrays(preds, method=thresholdType, threshold=binThreshold)

    if len(binaryPreds.shape) == 1:
        binaryGts, binaryPreds = binaryGts.reshape(1, -1), binaryPreds.reshape(1, -1)
        gts, preds = gts.reshape(1, -1), preds.reshape(1, -1)

    ret = []
    # check if we are computing classification scores. This will reduce computations if ground truth vectors are
    # completely 0
    classScores, realScores = False, False
    for metric in metrics:
        if metric in ['fscore', 'prec', 'rec', 'auc']:
            classScores = True
        elif metric in ['cs']:
            realScores = True

    rng = np.random.default_rng(888)
    for binGt, binPred, gt, pred in zip(binaryGts, binaryPreds, gts, preds):
        mets = []
        emptyBGt, emptyBPred, emptyGt, emptyPred = _check_correct_array_values(binGt, binPred, gt, pred)

        i = rng.integers(0, len(binGt))
        if classScores:
            if emptyBGt:
                warnings.warn('Binary ground truth does not contain values != 0, so one entry is being flipped to 1 '
                              'for the metrics to be defined.')
                binGt[i] = 1
            if emptyBPred:
                warnings.warn('Binary prediction does not contain values != 0, so one entry is being flipped to 1 for '
                              'the metrics to be defined.')
                binPred[i] = 1
        if realScores:
            if emptyGt:
                warnings.warn('Ground truth does not contain values != 0, so 1e-4 is being added to one random entry '
                              'in both.')
                gt[i] += 1e-4
            if emptyPred:
                warnings.warn('Prediction does not contain values != 0, so 1e-4 is being added to one random entry '
                              'in both.')
                pred[i] += 1e-4

        for metric in metrics:
            mets.append(_individual_fi_metrics(gt, pred, binGt, binPred, metric, predsNegative, thresholdType))

        ret.append(mets)

    ret = np.array(ret).astype(np.float32)

    if average is True and binaryPreds.shape[0] > 1:
        ret = np.mean(ret, axis=0)
    elif binaryPreds.shape[0] == 1:
        return ret.squeeze()

    return ret


def cosine_similarity(u, v, bounding: str = 'abs') -> float:
    """
    Computes cosine similarity between two real valued arrays. If negative, returns 0.

    :param u: (array-like), real valued array of dimension n.
    :param v: (array-like), real valued array of dimension n.
    :param bounding: if the CS is < 0, bound it in [0, 1] via absolute value ('abs') or max(0, value) ('max')
    :return: (0, 1) cosine similarity. """

    dist = 1 - cdist([u], [v], metric='cosine')[0][0]
    if bounding == 'abs':
        return np.abs(dist)
    elif bounding == 'max':
        return max(0, dist)
    else:
        raise ValueError('bounding method not valid.')


# ===================================

def _main_fi():
    print('MANUALLY GENERATING EXPLANATIONS FROM THE LINEAR CLASSIFIER')
    m = TransparentLinearClassifier(randomState=1)
    feats = ['a', 'b', 'c', 'd']
    X, y = m.fit(featureNames=feats, nSamples=100)

    exps = m.explain(X[:2], y[:2])

    print('Predicting probabilities:')
    print(m.predict_proba(X[:2]))

    print('Generated explanations:')
    print(exps)

    print('Generated expression:')
    print(m.expression)
    print('Computed derivatives:')
    print(m.derivatives)

    print('\nCREATING THE ARTIFICIAL DATA AND EXPLANATIONS FROM THE API:')
    X, y, ex, fNames, model = gen_data_fi(nSamples=50, returnModel=True)
    print('Observation: ', X[1], y[1], 'Feature names: ', fNames)
    print('Feature importance: ', ex[1])


if __name__ == '__main__':
    _main_fi()
