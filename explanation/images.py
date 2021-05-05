from math import sqrt

import torch
from captum._utils.models import SkLearnLinearModel
from captum.attr._core.lime import get_exp_kernel_similarity_function, LimeBase

from explanation.featureImportance import _lime_perturb_func, _lime_identity


def lime_image_attributions(model, data, labelsToExplain):
    rbf_kernel = get_exp_kernel_similarity_function('euclidean', kernel_width=0.75 * sqrt(len(data[0])))

    limeAttr = LimeBase(model,
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


def kshap_image_attributions(model, data, labelsToExplain):
    pass


def torch_pixel_attributions(model, data, labelsToExplain, method='lime', randomState=888):
    """ Get model attributions as feature importance vectors for torch models working on tabular data.
    :param model: (object) trained Pytorch model.
    :param data: (Tensor) data for which to get attributions.
    :param labelsToExplain: (Tensor) class labels the attributions will be computed w.r.t.
    :param method: (str) attribution method: 'lime', 'shap'.
    :param randomState: (int) random seed.
    :return: (ndarray) normalized [-1, 1] observation feature attributions
    """
    # todo normalize
    torch.manual_seed(randomState)
    if method == 'lime':
        return lime_image_attributions(model, data, labelsToExplain)
    elif method == 'shap':
        return kshap_image_attributions(model, data, labelsToExplain)
    else:
        pass