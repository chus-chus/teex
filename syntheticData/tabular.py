""" Module for generation of synthetic tabular data and ground truth feature importance explanations. """

from sklearn.datasets import make_classification

from transparentModels.decisionRule import RuleClassifier
from transparentModels.featureImportance import LinearClassifier
from utils.rule import rule_to_feature_importance
from utils.tabular import generate_feature_names


def gen_tabular_data(nSamples: int = 1000, nFeatures: int = 3, randomState: int = 888, method='rule',
                     returnModel=False, featureNames=None):
    """ Generate synthetic classification tabular data with ground truth explanations as feature importance vectors.

    :param nSamples: (int) number of samples in the data.
    :param nFeatures: (int) total number of features.
    :param randomState: (int) random seed.
    :param method: (str, ['rule', 'linear']) how the explanations are generated. If None, no explanations are
                   computed. Available: 'rule' (vectors will be binary), 'linear' (vectors will be real-valued)
    :param returnModel: (bool) should the model used for the generation of the explanations be returned?
    :param featureNames: (array-like) names of the generated features.
    :return: (ndarrays) X, y, explanations, featureNames (if they were not specified),
                        model (optional, param 'returnModel').
    """

    retFNames = False
    if featureNames is None:
        retFNames = True
        featureNames = generate_feature_names(nFeatures)

    if method == 'rule':
        data, targets = make_classification(n_samples=nSamples, n_classes=2, n_features=nFeatures,
                                            n_informative=nFeatures, n_redundant=0, n_repeated=0,
                                            random_state=randomState)
        # todo add randomness
        classifier = RuleClassifier(random_state=randomState)
        classifier.fit(data, targets, featureNames=featureNames)
        explanations = rule_to_feature_importance(classifier.explain(data), classifier.featureNames)
    elif method == 'linear':
        classifier = LinearClassifier(randomState=randomState)
        data, targets = classifier.fit(nSamples=nSamples, featureNames=featureNames)
        explanations = classifier.explain(data, newLabels=targets)
    else:
        raise ValueError(f"Explanation method not valid. Use {['rule', 'linear']}")

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


if __name__ == '__main__':

    print('AUTOMATICALLY GENERATING RULE EXPLANATIONS.')
    # X, y = gen_tabular_data(explanations=None)
    # print('\n', X[1:3], y[1:3])
    # X, y, ex = gen_tabular_data()
    # print('\n', X[1:3], y[1:3], ex[1:3])
    X, y, ex, fNames, model = gen_tabular_data(returnModel=True, nSamples=50, method='rule')
    print('Observation: ', X[1], y[1], 'Names: ', fNames)
    print('\nRule: ', model.explain(X[1].reshape(1, -1))[0], 'Feature importance: ', ex[1])

    # print('\nONE CAN ALSO EXTRACT THE ORIGINAL RULE EXPLANATIONS FROM THE MODEL:')
    # explan = model.explain(X[1:3])
    # for e in explan:
    #     print(e)

    X, y, ex, fNames, model = gen_tabular_data(returnModel=True, nSamples=50, method='linear')
    print('Observation: ', X[1], y[1], 'Names: ', fNames)
    print('\nFeature importance: ', ex[1])

