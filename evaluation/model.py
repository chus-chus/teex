""" Module for automatic evaluation of explanators on arbitrary models. """

import sys

import numpy as np

from sklearn.model_selection import train_test_split

from explanation.featureImportance import lime_torch_attributions, torch_tab_attributions
from explanation.images import torch_pixel_attributions
from syntheticData.image import gen_image_data
from syntheticData.tabular import gen_tabular_data


def _init_sk_model(classname, modelParams):
    """ Initialise sklearn model from string given the dict of params 'modelParams'. For it to work the model
    class needs to be imported in the context. """

    model = getattr(sys.modules[__name__], classname)
    return model(**modelParams)


def _gen_split_data(dataType, nSamples, nFeatures, randomState, expType, dataSplit):
    """ Returns train val test split synthetic image or tabular data """

    if dataType == 'image':
        # todo further parameters to gen_image_data
        # in the case of images, the binary g.t. masks are the explanations
        X, y, _ = gen_image_data(nSamples=nSamples, randomState=randomState)
    elif dataType == 'tab':
        X, y, gtExp, featureNames = gen_tabular_data(nSamples, nFeatures, randomState, expType)
    else:
        raise ValueError('DataType not valid.')

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

    if dataType == 'image':
        XTrain, XTest, yTrain, yTest = train_test_split(X, y, train_size=trainSize, random_state=randomState)
        XTrain, XVal, yTrain, yVal = train_test_split(XTrain, yTrain, test_size=valSize, random_state=randomState)
        return XTrain, XVal, XTest, yTrain, yVal, yTest
    elif dataType == 'tab':
        XTrain, XTest, yTrain, yTest, gtExpTrain, gtExpTest = train_test_split(X, y, gtExp, train_size=trainSize,
                                                                               random_state=randomState)
        XTrain, XVal, yTrain, yVal, gtExpTrain, gtExpVal = train_test_split(XTrain, yTrain, gtExpTrain, test_size=valSize,
                                                                            random_state=randomState)
        return XTrain, XVal, XTest, yTrain, yVal, yTest, gtExpTrain, gtExpVal, gtExpTest, featureNames


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

    XTrain = torch.FloatTensor(XTrain)
    XVal = torch.FloatTensor(XVal)
    XTest = torch.FloatTensor(XTest)
    gtExpTrain = torch.FloatTensor(gtExpTrain)
    gtExpVal = torch.FloatTensor(gtExpVal)
    gtExpTest = torch.FloatTensor(gtExpTest)

    yTrain = torch.LongTensor(yTrain)
    yVal = torch.LongTensor(yVal)
    yTest = torch.LongTensor(yTest)

    model = trainFunction(model, XTrain, yTrain)

    expTrain = torch_tab_attributions(model, XTrain, yTrain, method=expMethod)
    expVal = torch_tab_attributions(model, XVal, yVal, method=expMethod)
    expTest = torch_tab_attributions(model, XTest, yTest, method=expMethod)

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
    XTrain, XVal, XTest, yTrain, yVal, yTest = _gen_split_data('image', nSamples, nFeatures, randomState, expType,
                                                               dataSplit)

    XTrain = torch.FloatTensor(XTrain).reshape(-1, 3, 32, 32)
    XVal = torch.FloatTensor(XVal).reshape(-1, 3, 32, 32)
    XTest = torch.FloatTensor(XTest).reshape(-1, 3, 32, 32)

    yTrain = torch.LongTensor(yTrain.astype(int)).reshape(-1, 1, 32, 32)
    yVal = torch.LongTensor(yVal.astype(int)).reshape(-1, 1, 32, 32)
    yTest = torch.LongTensor(yTest.astype(int)).reshape(-1, 1, 32, 32)

    model = trainFunction(model, XTrain, yTrain)

    expTrain = torch_pixel_attributions(model, XTrain, method=expMethod)
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

    class TabNet(nn.Module):
        """ Basic FC NN """

        def __init__(self):
            super(TabNet, self).__init__()
            self.fc1 = nn.Linear(nFeat, 25)  # 5*5 from image dimension
            self.fc2 = nn.Linear(25, 15)
            self.fc3 = nn.Linear(15, 2)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    # sample training function for class Net
    def train_tabnet(model, X, y):
        """ X: FloatTensor, y: LongTensor """
        batchSize = 20
        nEpochs = 10
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=1e-3)
        for epoch in range(nEpochs):
            for batch in range(int(len(X) / batchSize)):
                XBatch = X[batch:batch + batchSize]
                yBatch = y[batch:batch + batchSize]
                out = model(XBatch)
                loss = criterion(out, yBatch)

                model.zero_grad()
                loss.backward()
                optimizer.step()
        return model


    class ImNet(nn.Module):
        # https://medium.com/analytics-vidhya/creating-a-very-simple-u-net-model-with-pytorch-for-semantic-segmentation-of-satellite-images-223aa216e705
        def __init__(self, in_channels, out_channels):
            super().__init__()

            self.conv1 = self.contract_block(in_channels, 32, 7, 3)
            self.conv2 = self.contract_block(32, 64, 3, 1)
            self.conv3 = self.contract_block(64, 128, 3, 1)

            self.upconv3 = self.expand_block(128, 64, 3, 1)
            self.upconv2 = self.expand_block(64 * 2, 32, 3, 1)
            self.upconv1 = self.expand_block(32 * 2, out_channels, 3, 1)

        def forward(self, x):
            # downsampling part
            conv1 = self.conv1(x)
            conv2 = self.conv2(conv1)
            conv3 = self.conv3(conv2)

            upconv3 = self.upconv3(conv3)

            upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
            upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

            return upconv1

        @staticmethod
        def contract_block(in_channels, out_channels, kernel_size, padding):
            contract = nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU(),
                torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )

            return contract

        @staticmethod
        def expand_block(in_channels, out_channels, kernel_size, padding):
            expand = nn.Sequential(torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
                                   torch.nn.BatchNorm2d(out_channels),
                                   torch.nn.ReLU(),
                                   torch.nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
                                   torch.nn.BatchNorm2d(out_channels),
                                   torch.nn.ReLU(),
                                   torch.nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2,
                                                            padding=1, output_padding=1)
                                   )
            return expand

    # sample training function for class Net
    def train_imnet(model, X, y):
        # todo y FloatTensor
        """ X: FloatTensor, y: FloatTensor """
        batchSize = 20
        nEpochs = 10
        criterion = nn.BCELoss()
        optimizer = optim.SGD(model.parameters(), lr=1e-3)
        for epoch in range(nEpochs):
            for batch in range(int(len(X) / batchSize)):
                XBatch = X[batch:batch + batchSize]
                yBatch = y[batch:batch + batchSize]
                out = F.softmax(model(XBatch), dim=1)
                loss = criterion(out, yBatch)

                model.zero_grad()
                loss.backward()
                optimizer.step()
        return model


    eval_torch_image(ImNet(3, 1), train_tabnet, 50, nFeat, (0.8, 0.1, 0.1), 'shap', 'rule')

    # eval_torch_tab(Net(), train_tabnet, 500, nFeat, (0.8, 0.1, 0.1), 'shap', 'rule')

    # eval_sk_tabular(DecisionTreeClassifier(), 1000, 5, (0.7, 0.15, 0.15), 'lime', 'rule')
