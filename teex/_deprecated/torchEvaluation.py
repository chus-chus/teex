""" Module for automatic evaluation of explanators on pytorch models. """

import numpy as np

from sklearn.model_selection import train_test_split

from teex._utils._explanation.featureImportance import torch_tab_attributions
from teex._utils._explanation.images import torch_pixel_attributions
from teex.featureImportance import feature_importance_scores, gen_data_fi
from teex.saliencyMap import saliency_map_scores, gen_data_sm
from teex._utils._misc import deprecated

import torch

# todo merge functions eval_torch_tab, eval_torch_image and their dependencies


# def _init_sk_model(classname, modelParams):
#     """ Initialise sklearn model from string given the dict of params 'modelParams'. For it to work the model
#     class needs to be imported in the context. """
#
#     model = getattr(sys.modules[__name__], classname)
#     return model(**modelParams)


@deprecated
def gen_split_data(dataType, nSamples, nFeatures, randomState, expType, dataSplit, **kwargs):
    """ Returns train val test split synthetic image or tabular data
    'kwargs' is passed to gen_image_data or gen_tabular_data, depending on 'dataType'.  """

    if dataType == 'image':
        X, y, gtExp, _ = gen_data_sm(nSamples=nSamples, randomState=randomState, **kwargs)
    elif dataType == 'tab':
        X, y, gtExp, featureNames = gen_data_fi(nSamples, nFeatures, randomState, expType, **kwargs)
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

    XTrain, XTest, yTrain, yTest, gtExpTrain, gtExpTest = train_test_split(X, y, gtExp, train_size=trainSize,
                                                                           random_state=randomState)
    XTrain, XVal, yTrain, yVal, gtExpTrain, gtExpVal = train_test_split(XTrain, yTrain, gtExpTrain, test_size=valSize,
                                                                        random_state=randomState)
    if dataType == 'image':
        return XTrain, XVal, XTest, yTrain, yVal, yTest, gtExpTrain, gtExpVal, gtExpTest
    else:
        return XTrain, XVal, XTest, yTrain, yVal, yTest, gtExpTrain, gtExpVal, gtExpTest, featureNames


@deprecated
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
    gtExpTrain, gtExpVal, gtExpTest, _ = gen_split_data('tab', nSamples, randomState, expType, dataSplit, nFeatures)

    model.fit(XTrain, yTrain)

    # expTrain = gen_explanations(model, XTrain, method=expMethod)
    # expVal = gen_explanations(model, XVal, method=expMethod)
    # expTest = gen_explanations(model, XTest, method=expMethod)

    return


@deprecated
def eval_torch_tab(model, trainFunction, nSamples, nFeatures, dataSplit, expMethod, expType, positiveClassLabel=1,
                   metrics=None, randomState=888, **kwargs):
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
    :param positiveClassLabel: internal representation of the positive class.
    :param metrics: (array-like, optional) quality metrics to compute. Available ['fscore', 'prec', 'rec', 'cs']
    :param randomState: (int) random seed.
    :param kwargs: extra arguments, will be passed to get_tabular_data
    """
    XTrain, XVal, XTest, yTrain, yVal, yTest, \
        gtExpTrain, gtExpVal, gtExpTest, fNames = gen_split_data(dataType='tab', nSamples=nSamples, nFeatures=nFeatures,
                                                                 randomState=randomState, expType=expType,
                                                                 dataSplit=dataSplit, **kwargs)
    XTrain = torch.FloatTensor(XTrain)
    XVal = torch.FloatTensor(XVal)
    XTest = torch.FloatTensor(XTest)

    yTrain = torch.LongTensor(yTrain)
    yVal = torch.LongTensor(yVal)
    yTest = torch.LongTensor(yTest)

    # todo add metrics for test, remove validation?
    model, avgLoss, avgAcc = trainFunction(model, XTrain, yTrain)

    expTrain = torch_tab_attributions(model, XTrain[yTrain == positiveClassLabel], yTrain, method=expMethod)
    expVal = torch_tab_attributions(model, XVal[yVal == positiveClassLabel], yVal, method=expMethod)
    expTest = torch_tab_attributions(model, XTest[yTest == positiveClassLabel], yTest, method=expMethod)

    # evaluate explanations
    if metrics is None:
        metrics = ['fscore', 'prec', 'rec', 'cs']

    expTrainScores = np.array([feature_importance_scores(gtExpTrain[np.where(yTrain == positiveClassLabel)][i], exp,
                                                         metrics=metrics) for i, exp in enumerate(expTrain)])
    expValScores = np.array([feature_importance_scores(gtExpVal[np.where(yVal == positiveClassLabel)][i], exp,
                                                       metrics=metrics) for i, exp in enumerate(expVal)])
    expTestScores = np.array([feature_importance_scores(gtExpTest[np.where(yTest == positiveClassLabel)][i], exp,
                                                        metrics=metrics) for i, exp in enumerate(expTest)])

    return np.mean(expTrainScores, axis=0), np.mean(expValScores, axis=0), np.mean(expTestScores, axis=0)


@deprecated
def eval_torch_image(model, trainFunction, nSamples, dataSplit, expMethod, imageH=32, imageW=32, patternH=16,
                     patternW=16, cellH=4, cellW=4, patternProp=0.5, fillPct=0.4, randomState=888,
                     metrics=None, positiveClassLabel=1, **kwargs):
    """
    Trains a PyTorch model with synthetic image data. Then, generates explanations and evaluates them with
    the available ground truths for the train, validation and test sets. The model must be able to work with batches of
    shape [-1, 3, imageH, imageW]

    :param model: torch model instance.
    :param trainFunction: function to train the torch model with. As parameters, it must accept 'model' (the torch
                          model), 'X' (the training data as a ndarray)  and 'y' (g.t. labels as a ndarray) and returns
                          the trained model.
    :param nSamples: (int) number of samples to create.
    :param dataSplit: (array-like) list like object with 3 floats, each indicating the proportion of data samples
                                   to use as train, validation and test, respectively. Must sum 1. i.e. (0.6, 0.2, 0.2)
    # todo update supported methods
    :param expMethod: (string) explanation method to generate the explanations with. Available ['kernelShap', 'lime']
    :param imageH: (int) height in pixels of the images.
    :param imageW: (int) width in pixels of the images.
    :param patternH: (int) height in pixels of the pattern.
    :param patternW: (int) width in pixels of the pattern.
    :param cellH: (int) height in pixels of each cell.
    :param cellW: (int) width in pixels of each cell.
    :param patternProp: (float, [0, 1]) percentage of appearance of the pattern in the dataset.
    :param fillPct: (float, [0, 1]) percentage of cells filled (not black) in each image.
    :param metrics: (array-like, optional) quality metrics to compute. Available ['auc', 'fscore', 'prec', 'rec', 'cs']
    :param kwargs: extra arguments for the .attribute method of the selected explainer.
    :param positiveClassLabel: internal representation of the positive class in the labels set.
    :param randomState: (int) random seed.
    :return: mean metrics for the explanations of the positive class observations.
    """
    expType, nFeatures = None, None
    XTrain, XVal, XTest, yTrain, yVal, yTest, \
        gtExpTrain, gtExpVal, gtExpTest = gen_split_data('image', nSamples, nFeatures, randomState, expType, dataSplit,
                                                         imageH=imageH, imageW=imageW, patternH=patternH,
                                                         patternW=patternW, patternProp=patternProp, cellH=cellH,
                                                         cellW=cellW, fillPct=fillPct)

    XTrain = torch.FloatTensor(XTrain).reshape(-1, 3, imageH, imageW)
    XVal = torch.FloatTensor(XVal).reshape(-1, 3, imageH, imageW)
    XTest = torch.FloatTensor(XTest).reshape(-1, 3, imageH, imageW)

    yTrain = torch.LongTensor(yTrain)
    yVal = torch.LongTensor(yVal)
    yTest = torch.LongTensor(yTest)

    model, avgLoss, avgAcc = trainFunction(model, XTrain, yTrain, XVal, yVal)

    print(f'Trained model with avg. training loss of {avgLoss} and avg. training accuracy of {avgAcc}.')
    print(f'Generating explanations with {expMethod}')

    # explanations only for the positive class
    expTrain = torch_pixel_attributions(model, XTrain[yTrain == positiveClassLabel], yTrain, method=expMethod, **kwargs)
    expVal = torch_pixel_attributions(model, XVal[yVal == positiveClassLabel], yVal, method=expMethod, **kwargs)
    expTest = torch_pixel_attributions(model, XTest[yTest == positiveClassLabel], yTest, method=expMethod, **kwargs)

    # evaluate explanations
    if metrics is None:
        metrics = ['auc', 'fscore', 'prec', 'rec', 'cs']

    expTrainScores = np.array([saliency_map_scores(gtExpTrain[np.where(yTrain == positiveClassLabel)][i], exp,
                                                   metrics=metrics)
                               for i, exp in enumerate(expTrain)])
    expValScores = np.array([saliency_map_scores(gtExpVal[np.where(yVal == positiveClassLabel)][i], exp,
                                                 metrics=metrics) for i, exp in enumerate(expVal)])
    expTestScores = np.array([saliency_map_scores(gtExpTest[np.where(yTest == positiveClassLabel)][i], exp,
                                                  metrics=metrics) for i, exp in enumerate(expTest)])

    return np.mean(expTrainScores, axis=0), np.mean(expValScores, axis=0), np.mean(expTestScores, axis=0)


def _model_eval_main():

    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    class FCNN(nn.Module):
        """ Basic NN for image classification. """

        def __init__(self, imH, imW, cellH):
            super(FCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 3, kernel_size=cellH, stride=1)
            self.H1, self.W1 = imH - 3, imW - 3
            self.conv2 = nn.Conv2d(3, 1, kernel_size=cellH, stride=1)
            self.H2, self.W2 = self.H1 - 3, self.W1 - 3
            self.fc1 = nn.Linear(self.H2 * self.W2, 100)
            self.fc2 = nn.Linear(100, 20)
            self.fc3 = nn.Linear(20, 2)

        def forward(self, x):
            if len(x.shape) == 3:
                # single instance, add the batch dimension
                print(x.shape)
                x = x.view(-1, x.shape[2], x.shape[1], x.shape[0])

            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.fc1(x.view(x.shape[0], -1)))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    # sample training function for binary classification
    def train_net(model, X, y, XVal, yVal, randomState=888):
        """ X: FloatTensor, y: LongTensor """
        torch.manual_seed(randomState)
        batchSize = 50
        nEpochs = 100
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        bestValAcc = -np.inf
        for epoch in range(nEpochs):
            model.train()
            for batch in range(int(len(X) / batchSize)):
                XBatch = X[batch:batch + batchSize]
                yBatch = y[batch:batch + batchSize]
                model.zero_grad()
                out = model(XBatch)
                loss = criterion(out, yBatch)
                loss.backward()
                optimizer.step()
            model.eval()
            avgLoss = 0
            avgAcc = 0
            for batch in range(int(len(X) / batchSize)):
                XBatch = X[batch:batch + batchSize]
                yBatch = y[batch:batch + batchSize]
                out = model(XBatch)
                loss = criterion(out, yBatch)
                avgLoss += loss.item()
                preds = F.softmax(out, dim=-1).argmax(dim=1)
                acc = accuracy_score(yBatch, preds.detach().numpy())
                avgAcc += acc

            avgValLoss = 0
            avgValAcc = 0
            for batch in range(int(len(XVal) / batchSize)):
                XBatch = XVal[batch:batch + batchSize]
                yBatch = yVal[batch:batch + batchSize]
                out = model(XBatch)
                loss = criterion(out, yBatch)
                avgValLoss += loss.item()
                preds = F.softmax(out, dim=-1).argmax(dim=1)
                acc = accuracy_score(yBatch, preds.detach().numpy())
                avgValAcc += acc

            avgValAcc /= int(len(XVal) / batchSize)
            avgLoss /= int(len(X) / batchSize)
            avgAcc /= int(len(X) / batchSize)

            # early stoppage
            if avgValAcc >= bestValAcc:
                bestValAcc = avgValAcc
            else:
                break

        return model, avgLoss, avgAcc

    nSamples = 100
    randomState = 8
    imageH, imageW = 32, 32
    patternH, patternW = 16, 16
    cellH, cellW = 4, 4
    patternProp = 0.5
    fillPct = 0.4
    colorDev = 0.1

    X, y, exps, pat = gen_data_sm(nSamples=1000, imageH=imageH, imageW=imageW, patternH=patternH, patternW=patternW,
                                  cellH=cellH, cellW=cellW, patternProp=patternProp, fillPct=fillPct, colorDev=0.5,
                                  randomState=7)

    XTrain, XTest, yTrain, yTest, expsTrain, expsTest = train_test_split(X, y, exps, train_size=0.8, random_state=7)
    XTrain, XVal, yTrain, yVal, expsTrain, expsVal = train_test_split(XTrain, yTrain, expsTrain, train_size=0.6,
                                                                      random_state=7)

    XTrain = torch.FloatTensor(XTrain).permute(0, 3, 2, 1)
    yTrain = torch.LongTensor(yTrain)
    XVal = torch.FloatTensor(XVal).permute(0, 3, 2, 1)
    yVal = torch.LongTensor(yVal)
    XTest = torch.FloatTensor(XTest).permute(0, 3, 2, 1)

    nFeatures = len(XTrain[0].flatten())
    model, trLoss, trAcc = train_net(FCNN(imageH, imageW, cellH), XTrain, yTrain, XVal, yVal, randomState=7)

    print(f'Tr. loss: {round(trLoss, 3)}, Tr. accuracy: {round(trAcc, 3)}')
    print(
        f'Validation accuracy: {round(accuracy_score(yVal, F.softmax(model(torch.FloatTensor(XVal)), dim=-1).argmax(dim=1).detach().numpy()), 3)}')
    print(
        f'Test accuracy: {round(accuracy_score(yTest, F.softmax(model(torch.FloatTensor(XTest)), dim=-1).argmax(dim=1).detach().numpy()), 3)}')

    expMethod = 'integratedGradient'
    genExpsTrain = torch_pixel_attributions(model, XTrain[yTrain == 1], yTrain[yTrain == 1], method=expMethod)

    # ----------------------------------------------------------------------------------------------------
    class FCNN(nn.Module):
        """ Basic FC NN """

        def __init__(self, nFeatures):
            super(FCNN, self).__init__()
            self.nFeatures = nFeatures
            self.fc1 = nn.Linear(nFeatures, 25)  # 5*5 from image dimension
            self.fc2 = nn.Linear(25, 15)
            self.fc3 = nn.Linear(15, 2)

        def forward(self, x):
            x = x.reshape(-1, self.nFeatures)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    class ImNet(nn.Module):
        # https://medium.com/analytics-vidhya/creating-a-very-simple-u-net-model-with-pytorch-for-semantic-segmentation-of-satellite-images-223aa216e705
        def __init__(self, in_channels, out_channels, imH, imW):
            super().__init__()

            self.conv1 = self.contract_block(in_channels, 32, 7, 3)
            self.conv2 = self.contract_block(32, 64, 3, 1)
            self.conv3 = self.contract_block(64, 128, 3, 1)

            self.upconv3 = self.expand_block(128, 64, 3, 1)
            self.upconv2 = self.expand_block(64 * 2, 32, 3, 1)
            self.upconv1 = self.expand_block(32 * 2, out_channels, 3, 1)

            self.linear = nn.Linear(out_channels * imH * imW, 2)

        def forward(self, x):
            # downsampling
            conv1 = self.conv1(x)
            conv2 = self.conv2(conv1)
            conv3 = self.conv3(conv2)

            # upsampling
            upconv3 = self.upconv3(conv3)
            upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
            upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

            linOut = self.linear(upconv1.reshape(upconv1.shape[0], -1))

            return linOut

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

    from sklearn.metrics import accuracy_score

    # sample training function for binary classification
    def train_net(model, X, y, randomState=888):
        """ X: FloatTensor, y: LongTensor """
        torch.manual_seed(randomState)
        batchSize = 20
        nEpochs = 10
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        avgLoss, avgAcc = None, None
        for epoch in range(nEpochs):
            model.train()
            for batch in range(int(len(X) / batchSize)):
                XBatch = X[batch:batch + batchSize]
                yBatch = y[batch:batch + batchSize]
                model.zero_grad()
                out = model(XBatch)
                loss = criterion(out, yBatch)

                loss.backward()
                optimizer.step()
            model.eval()
            avgLoss = 0
            avgAcc = 0
            for batch in range(int(len(X) / batchSize)):
                XBatch = X[batch:batch + batchSize]
                yBatch = y[batch:batch + batchSize]
                out = model(XBatch)
                loss = criterion(out, yBatch)
                avgLoss += loss.item()
                preds = F.softmax(out, dim=-1).argmax(dim=1)
                acc = accuracy_score(yBatch, preds.detach().numpy())
                avgAcc += acc
            avgLoss /= int(len(X) / batchSize)
            avgAcc /= int(len(X) / batchSize)

        return model, avgLoss, avgAcc

    imH = 32
    imW = 32
    pattH = 16
    pattW = 16
    cH = 4
    cW = 4

    # image Umodel with pixel importance explanations
    # nImages = 100
    # torch.manual_seed(888)
    # imModel = FCNN(imH * imW * 3)
    # # print('Training image model...')
    # eval_torch_image(model=imModel,
    #                  trainFunction=train_net,
    #                  nSamples=nImages,
    #                  imageH=imH,
    #                  imageW=imW,
    #                  patternH=pattH,
    #                  patternW=pattW,
    #                  cellH=cH,
    #                  cellW=cW,
    #                  dataSplit=(0.8, 0.1, 0.1),
    #                  expMethod='integratedGradient')

    # tabular model with feature importance explanations
    nFeat = 5
    torch.manual_seed(888)
    tabModel = FCNN(nFeat)
    eval_torch_tab(model=tabModel,
                   trainFunction=train_net,
                   nSamples=200,
                   nFeatures=nFeat,
                   dataSplit=(0.8, 0.1, 0.1),
                   expType='fi',
                   expMethod='gradientShap')
    # todo individual instance evaluation


if __name__ == '__main__':
    _model_eval_main()
