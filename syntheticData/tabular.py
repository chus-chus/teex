""" Module for generation of synthetic tabular data and ground truth feature importance explanations. """

from sklearn.datasets import make_classification

from transparentModels.decisionRule import RuleClassifier
from utils.rule import rule_to_feature_importance


def gen_tabular_data(nSamples: int = 1000, nClasses: int = 2, nFeatures: int = 3, nInformative: int = None,
                     nRedundant: int = None, randomState: int = 888, explanations='rule', returnModel=False):
    """ Generate synthetic classification tabular data with ground truth explanations as feature importance vectors.

    :param nSamples: number of samples in the data
    :param nClasses: numer of classes in the data
    :param nFeatures: total number of features
    :param nInformative: number of informative features
    :param nRedundant: number of redundant features
    :param randomState: random seed
    :param explanations: how the explanations are generated. If None, no explanations are computed.
                         Available: 'rule' (vectors will be binary), 'linear' (vectors will be real-valued)
    :param returnModel: should the model used for the generation of the explanations be returned?
                        (only if 'explanations' != None)
    :return: arrays X, y, explanations (optional), model (optional)
    """
    if nInformative is None and nRedundant is not None:
        nInformative = nFeatures - nRedundant
    elif nRedundant is None and nInformative is not None:
        nRedundant = nFeatures - nInformative
    else:
        nInformative = nFeatures
        nRedundant = 0

    if nClasses != 2 and explanations == 'linear':
        raise ValueError('Linear explanations can only be generated with two classes.')

    data, targets = make_classification(n_samples=nSamples, n_classes=nClasses, n_features=nFeatures,
                                        n_informative=nInformative, n_redundant=nRedundant, random_state=randomState)
    # todo add randomness

    if explanations is None:
        if returnModel:
            raise ValueError('Cannot return model without generating explanations.')
        else:
            return data, targets

    if explanations == 'rule':
        classifier = RuleClassifier(random_state=randomState)
        classifier.fit(data, targets)
        explanations = rule_to_feature_importance(classifier.explain(data), classifier.featureNames)
    elif explanations == 'linear':
        raise NotImplementedError
    else:
        raise ValueError(f"Explanation method not valid. Use {['rule', 'linear']}")

    if returnModel:
        return data, targets, explanations, classifier
    else:
        return data, targets, explanations


if __name__ == '__main__':

    print('AUTOMATICALLY GENERATING RULE EXPLANATIONS.')
    # X, y = gen_tabular_data(explanations=None)
    # print('\n', X[1:3], y[1:3])
    # X, y, ex = gen_tabular_data()
    # print('\n', X[1:3], y[1:3], ex[1:3])
    X, y, ex, model = gen_tabular_data(returnModel=True)
    print('Observation: ', X[1])
    print('\n Rule: ', model.explain(X[1].reshape(1, -1))[0], 'Feature importance: ', ex[1])

    # print('\nONE CAN ALSO EXTRACT THE ORIGINAL RULE EXPLANATIONS FROM THE MODEL:')
    # explan = model.explain(X[1:3])
    # for e in explan:
    #     print(e)
