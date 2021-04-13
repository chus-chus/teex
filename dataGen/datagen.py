""" Module for generation of synthetic datasets with available ground truths. """
from sklearn.datasets import make_classification


def gen_tabular_data(nSamples: int = 1000, nClasses: int = 2, nFeatures: int = 3, nInformative: int = 2,
                     randomState: int = 888):
    data, targets = make_classification(n_samples=nSamples, n_classes=nClasses, n_features=nFeatures,
                                        n_informative=nInformative, random_state=randomState)
    # todo add randomness
    return data, targets

