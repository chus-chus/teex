""" Module for generation of synthetic datasets with available ground truths. """
from sklearn.datasets import make_classification


def gen_tabular_data(nSamples: int = 1000, nClasses: int = 2, nFeatures: int = 3, nInformative: int = None,
                     nRedundant: int = None, randomState: int = 888):
    if nInformative is None and nRedundant is not None:
        nInformative = nFeatures - nRedundant
    elif nRedundant is None and nInformative is not None:
        nRedundant = nFeatures - nInformative
    else:
        nInformative = nFeatures
        nRedundant = 0
    data, targets = make_classification(n_samples=nSamples, n_classes=nClasses, n_features=nFeatures,
                                        n_informative=nInformative, n_redundant=nRedundant, random_state=randomState)
    # todo add randomness
    return data, targets

