from math import sqrt

import numpy as np
import torch

from captum._utils.models import SkLearnLinearModel
from captum.attr import KernelShap
# from captum.attr import LimeBase
# from captum.attr._core.lime import get_exp_kernel_similarity_function

# from TAIAOexp.utils._explanation.featureImportance import _lime_perturb_func, _lime_identity
from TAIAOexp.utils._arrays import _minmax_normalize_array
from TAIAOexp.utils._explanation.general import get_explainer, get_attributions


# def lime_image_attributions(model, data, labelsToExplain):
#     rbf_kernel = get_exp_kernel_similarity_function('euclidean', kernel_width=0.75 * sqrt(len(data[0])))
#
#     limeAttr = LimeBase(model,
#                         SkLearnLinearModel("linear_model.Ridge"),
#                         similarity_func=rbf_kernel,
#                         perturb_func=_lime_perturb_func,
#                         perturb_interpretable_space=False,
#                         from_interp_rep_transform=None,
#                         to_interp_rep_transform=_lime_identity)
#     if len(data.shape) != 4:
#         raise ValueError('Data should be 4D (nObs x channels x height x width).')
#
#     attr = torch.FloatTensor([])
#     for obs, label in zip(data, labelsToExplain):
#         attr = torch.cat((attr, limeAttr.attribute(obs.reshape(1, 3, 32, 32))))
#
#     return attr.detach().numpy()


def kshap_image_attributions(net, data, labelsToExplain):
    """ Compute Kernel SHAP attribution coefficients for PyTorch model trained with image data.
    https://captum.ai/api/kernel_shap.html
    :param net: (object) PyTorch trained model.
    :param data: (Tensor) tabular data observations to get the attributions for.
    :param labelsToExplain: (Tensor) class labels the attributions will be computed w.r.t.
    :return: (ndarray) observation feature attributions
    """
    shapAttr = KernelShap(net)

    if len(data.shape) != 4:
        raise ValueError('Data should be 4D (nObs x channels x height x width).')

    attr = torch.FloatTensor([])
    for obs, label in zip(data, labelsToExplain):
        attr = torch.cat((attr, shapAttr.attribute(obs.reshape(1, 3, 32, 32), baselines=torch.zeros(data[0].shape),
                                                   target=label)))

    return attr.detach().numpy()


def torch_pixel_attributions(data, labelsToExplain, explainer, randomState=888, **kwargs):
    """ Get model attributions as feature importance vectors for torch models working on tabular data. Note that the
    input images must be alredy be able to be forwarded to the model without any further transformations.

    :param data: (array-like) data for which to get attributions (each image must be a Tensor)
    :param labelsToExplain: (Tensor) class labels the attributions will be computed w.r.t.
    :param explainer: (captum.attr Explainer) declared explainer.
    :param randomState: (int) random seed.
    :param kwargs: extra keyword arguments for the .attribute method of the explainer.
    :return: (ndarray) normalized [0, 1] observation feature attributions
    """

    torch.manual_seed(randomState)

    attributions = []
    for image, target in zip(data, labelsToExplain):
        if len(image.shape) == 3:
            image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
        attr = get_attributions(explainer, obs=image, target=target, method=method, **kwargs)
        attr = attr.squeeze()
        if len(attr.shape) == 3:
            attr = np.transpose(attr.squeeze().cpu().detach().numpy(), (1, 2, 0))
            # mean pool channel attributions
            attr = np.mean(attr, axis=2)
            # attr = attr.mean(dim=2)
        elif len(attr.shape) != 2:
            raise ValueError(f'Attribution shape {attr.shape} is not valid.')

        # viz._normalize_image_attr(tmp, 'absolute_value', 10)
        attributions.append(_minmax_normalize_array(attr))

    return np.array(attributions)
