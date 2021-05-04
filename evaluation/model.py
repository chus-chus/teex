""" Module for automatic evaluation of explanators on arbitrary models. """

import sys

import numpy as np

from sklearn.model_selection import train_test_split

from explanation.featureImportance import lime_torch_attributions, torch_tab_attributions
from syntheticData.tabular import gen_tabular_data


def _init_sk_model(classname, modelParams):
    """ Initialise sklearn model from string given the dict of params 'modelParams'. For it to work the model
    class needs to be imported in the context. """

    model = getattr(sys.modules[__name__], classname)
    return model(**modelParams)


def _gen_split_data(dataType, nSamples, nFeatures, randomState, expType, dataSplit):

    if dataType == 'image':
        pass
    elif dataType == 'tab':
        X, y, gtExp, featureNames = gen_tabular_data(nSamples, nFeatures, randomState, expType)

    if not isinstance(dataSplit, (np.ndarray, list, tuple)):
        raise ValueError('dataSplit should be a list-like object.')
    elif len(dataSplit) != 3:
        raise ValueError('dataSplit should contain three elements.')
    total = 0
    for elem in dataSplit:
        total += elem
    if total != 1:
        raise ValueError('elements in dataSplit should sum to 1.')

    trainSize = round(len(y) * dataSplit[0])
    valSize = round(len(y) * dataSplit[1])
    XTrain, XTest, yTrain, yTest, gtExpTrain, gtExpTest = train_test_split(X, y, gtExp, train_size=trainSize,
                                                                           random_state=randomState)
    XTrain, XVal, yTrain, yVal, gtExpTrain, gtExpVal = train_test_split(XTrain, yTrain, gtExpTrain, test_size=valSize,
                                                                        random_state=randomState)
    return XTrain, XVal, XTest, yTrain, yVal, yTest, gtExpTrain, gtExpVal, gtExpTest


# todo should feature importances be normalised for the linear method?
def eval_sk_tabular(model, nSamples, nFeatures, dataSplit, expMethod, expType, randomState=888):
    """ Trains a sklearn model with synthetic data. Then, generates explanations and evaluates them with
    the available ground truths for the generated train, validation and test sets.

    :param model: sklearn model instance.
    :param nSamples: (int) number of samples to create.
    :param nFeatures: (int) number of features the artificial data will have.
    :param dataSplit: (array-like) list like object with 3 floats, each indicating the proportion of data samples
                                   to use as train, validation and test, respectively. Must sum 1. i.e. (0.6, 0.2, 0.2)
    :param expMethod: (string) explanation method to generate the explanations with. Available ['shap', 'lime']
    :param expType: (string) type of ground truth explanations. Available:
                             'fi' (feature importance vect.), 'rule' (DecisionRule objects)
    :param randomState: (int) random seed.
    """
    XTrain, XVal, XTest, yTrain, yVal, yTest, \
        gtExpTrain, gtExpVal, gtExpTest = _gen_split_data('tab', nSamples, nFeatures, randomState, expType, dataSplit)

    model.fit(XTrain, yTrain)

    # expTrain = gen_explanations(model, XTrain, method=expMethod)
    # expVal = gen_explanations(model, XVal, method=expMethod)
    # expTest = gen_explanations(model, XTest, method=expMethod)

    return


def eval_torch_tab(model, trainFunction, nSamples, nFeatures, dataSplit, expMethod, expType, randomState=888):
    """ Trains a PyTorch model with synthetic tabular data. Then, generates explanations and evaluates them with
    the available ground truths for the generated train, validation and test sets.

    :param model: torch model instance.
    :param trainFunction: function to train the torch model with. As parameters, it must accept 'model' (the torch
                          model), 'X' (the training data as a ndarray)  and 'y' (g.t. labels as a ndarray) and returns
                          the trained model.
    :param nSamples: (int) number of samples to create.
    :param nFeatures: (int) number of features the artificial data will have.
    :param dataSplit: (array-like) list like object with 3 floats, each indicating the proportion of data samples
                                   to use as train, validation and test, respectively. Must sum 1. i.e. (0.6, 0.2, 0.2)
    :param expMethod: (string) explanation method to generate the explanations with. Available ['shap', 'lime']
    :param expType: (string) type of ground truth explanations. Available:
                             'fi' (feature importance vect.), 'rule' (DecisionRule objects)
    :param randomState: (int) random seed.
    """
    XTrain, XVal, XTest, yTrain, yVal, yTest, \
        gtExpTrain, gtExpVal, gtExpTest = _gen_split_data('tab', nSamples, nFeatures, randomState, expType, dataSplit)

    model = trainFunction(model, XTrain, yTrain)

    expTrain = torch_tab_attributions(model, torch.FloatTensor(XTrain), torch.LongTensor(yTrain), method=expMethod)
    expVal = torch_tab_attributions(model, torch.FloatTensor(XVal), torch.LongTensor(yVal), method=expMethod)
    expTest = torch_tab_attributions(model, torch.FloatTensor(XTest), torch.LongTensor(yTest), method=expMethod)

    return


def eval_torch_image(model, trainFunction, nSamples, nFeatures, dataSplit, expMethod, expType, randomState=888):
    """
    Trains a PyTorch model with synthetic image data. Then, generates explanations and evaluates them with
    the available ground truths for the train, validation and test sets.

    :param model: torch model instance.
    :param trainFunction: function to train the torch model with. As parameters, it must accept 'model' (the torch
                          model), 'X' (the training data as a ndarray)  and 'y' (g.t. labels as a ndarray) and returns
                          the trained model.
    :param nSamples: (int) number of samples to create.
    :param nFeatures: (int) number of features the artificial data will have.
    :param dataSplit: (array-like) list like object with 3 floats, each indicating the proportion of data samples
                                   to use as train, validation and test, respectively. Must sum 1. i.e. (0.6, 0.2, 0.2)
    :param expMethod: (string) explanation method to generate the explanations with. Available ['shap', 'lime']
    :param expType: (string) type of ground truth explanations. Available:
                             'fi' (feature importance vect.), 'rule' (DecisionRule objects)
    :param randomState: (int) random seed.
    """
    XTrain, XVal, XTest, yTrain, yVal, yTest, \
        gtExpTrain, gtExpVal, gtExpTest = _gen_split_data('image', nSamples, nFeatures, randomState, expType, dataSplit)

    model.fit(XTrain, yTrain)

    # expTrain = gen_explanations(model, XTrain, method=expMethod)
    # expVal = gen_explanations(model, XVal, method=expMethod)
    # expTest = gen_explanations(model, XTest, method=expMethod)

    return


if __name__ == '__main__':
    from sklearn.tree import DecisionTreeClassifier

    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F

    nFeat = 5

    class Net(nn.Module):
        """ Basic FC NN """

        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(nFeat, 25)  # 5*5 from image dimension
            self.fc2 = nn.Linear(25, 15)
            self.fc3 = nn.Linear(15, 2)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    # sample training function for class Net
    def train_nn(model, X, y):
        batchSize = 20
        nEpochs = 10
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=1e-3)
        X = torch.FloatTensor(X)
        y = torch.LongTensor(y)
        for epoch in range(nEpochs):
            for batch in range(int(len(X) / batchSize)):
                XBatch = X[batch:batch+batchSize]
                yBatch = y[batch:batch+batchSize]
                out = model(XBatch)
                loss = criterion(out, yBatch)

                model.zero_grad()
                loss.backward()
                optimizer.step()
        return model

    eval_torch_tab(Net(), train_nn, 1000, nFeat, (0.8, 0.1, 0.1), 'shap', 'rule')

    eval_sk_tabular(DecisionTreeClassifier(), 1000, 5, (0.7, 0.15, 0.15), 'lime', 'rule')
