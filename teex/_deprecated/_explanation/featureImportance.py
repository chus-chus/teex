""" Generation of feature importance explanations. """

import numpy as np
from math import sqrt

import torch

from captum.attr import LimeBase, KernelShap
#from captum._utils.models.linear_model import SkLearnLinearModel
from captum.attr._core.lime import get_exp_kernel_similarity_function

from teex._utils._explanation.images import get_explainer, get_attributions

from teex._utils._arrays import _minmax_normalize_array


def _lime_perturb_func(originalInput, **kwargs):
    """ Perturb the original input to find neighbour samples. Note that the implementation is quite specific to the
    data used in the model. In our case, the synthetic tabular samples are generated from a normal dist. with scale=1.
    """
    return originalInput + torch.randn_like(originalInput)


def _lime_identity(input1, input2, **kwargs):
    """ Maps the input tensors to the interpretable space (identity in this case) with the signature expected from
     captum.attr.LimeBase"""
    return input1


def lime_torch_attributions(net, data, labelsToExplain):
    """ Compute LIME attribution coefficients for PyTorch model trained with tabular data.
    https://captum.ai/api/lime.html
    :param net: (object) PyTorch trained model.
    :param data: (Tensor) tabular data observations to get the attributions for.
    :param labelsToExplain: (Tensor) class labels the attributions will be computed w.r.t.
    :return: (ndarray) observation feature attributions """
    rbf_kernel = get_exp_kernel_similarity_function('euclidean', kernel_width=0.75 * sqrt(len(data[0])))

    limeAttr = LimeBase(net,
                        SkLearnLinearModel("linear_model.Ridge"),
                        similarity_func=rbf_kernel,
                        perturb_func=_lime_perturb_func,
                        perturb_interpretable_space=False,
                        from_interp_rep_transform=None,
                        to_interp_rep_transform=_lime_identity)
    if len(data.shape) != 2:
        raise ValueError('Data should be 2D (nObs x nFeatures).')

    attr = torch.FloatTensor([])
    for obs, label in zip(data, labelsToExplain):
        attr = torch.cat((attr, limeAttr.attribute(obs.reshape(1, -1), target=label)))

    return attr.detach().numpy()


def kshap_torch_attributions(net, data, labelsToExplain):
    """ Compute Kernel SHAP attribution coefficients for PyTorch model trained with tabular data.
    https://captum.ai/api/kernel_shap.html
    :param net: (object) PyTorch trained model.
    :param data: (Tensor) tabular data observations to get the attributions for.
    :param labelsToExplain: (Tensor) class labels the attributions will be computed w.r.t.
    :return: (ndarray) observation feature attributions
    """
    shapAttr = KernelShap(net)

    if len(data.shape) != 2:
        raise ValueError('Data should be 2D (nObs x nFeatures).')

    attr = torch.FloatTensor([])
    for obs, label in zip(data, labelsToExplain):
        attr = torch.cat((attr, shapAttr.attribute(obs.reshape(1, -1), baselines=torch.zeros(len(data[0])),
                                                   target=label)))

    return attr.detach().numpy()


def get_tabular_attributions(explainer, inputObs, target, method, param):
    pass


def torch_tab_attributions(model, data, labelsToExplain, method='lime', randomState=888):
    """ Get model attributions as feature importance vectors for torch models working on tabular data.
    :param model: (object) trained Pytorch model.
    :param data: (Tensor) data for which to get attributions.
    :param labelsToExplain: (Tensor) class labels the attributions will be computed w.r.t.
    :param method: (str) attribution method: 'lime', 'shap'.
    :param randomState: (int) random seed.
    :return: (ndarray) normalized [-1, 1] observation feature attributions
    """

    explainer = get_explainer(model, method=method)

    attributions = []
    for observation, target in zip(data, labelsToExplain):
        attr = get_attributions(explainer, obs=observation, target=target, method=method)
        # todo normalization -1, 1
        attributions.append(_minmax_normalize_array(attr.detach().numpy()))

    return np.array(attributions)
