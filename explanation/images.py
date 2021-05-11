from math import sqrt

import numpy as np
import torch
from captum._utils.models import SkLearnLinearModel
from captum.attr import KernelShap, IntegratedGradients, GradientShap, Occlusion, GuidedBackprop, DeepLift, \
    GuidedGradCam
from captum.attr._core.lime import get_exp_kernel_similarity_function, LimeBase, Lime

from explanation.featureImportance import _lime_perturb_func, _lime_identity
from utils.image import normalize_array


def get_image_explainer(model, layer=None, method='guidedGradCAM', model_name=None, **kwargs):
    """ Gets a captum explainer based on the provided PyTorch model.
    :param model: (callable) PyTorch classification model
    :param layer: (object) Model layer to use guidedGradCAM on. Predefined layer numbers are automatically defined if model_name
                  is in ['resnet18', 'resnet50', 'alexnet', 'vgg', 'squeezenet', 'densenet', 'inception'].
    :param method: (str) explanation method. Available: ['lime', 'integratedGradient', 'gradientShap', 'kernelShap',
    'occlusion', 'guidedBackProp', 'deepLift', 'guidedGradCAM']
    :param model_name: (str) Model architecture. Only used if layer=None and method='guidedGradCAM'.
                       Options: ['resnet18', 'resnet50', 'alexnet', 'vgg', 'squeezenet', 'densenet', 'inception']
    :param kwargs: extra keyword arguments. Will be passed to the explainer's __init__ method (for those that support
                   extra arguments).
    :return: explainer object.
    """
    if method == 'lime':
        explainer = Lime(model, **kwargs)
    elif method == 'integratedGradient':
        explainer = IntegratedGradients(model, **kwargs)
    elif method == 'gradientShap':
        explainer = GradientShap(model, **kwargs)
    elif method == 'kernelShap':
        explainer = KernelShap(model)
    elif method == 'occlusion':
        explainer = Occlusion(model)
    elif method == 'guidedBackProp':
        explainer = GuidedBackprop(model)
    elif method == 'deepLift':
        explainer = DeepLift(model, **kwargs)
    elif method == 'guidedGradCAM':
        if layer is None:
            if model_name == 'resnet18':
                layer = model.layer4
            elif model_name == 'resnet50':
                layer = model.layer4
            elif model_name == 'alexnet':
                layer = model.features[10]
            elif model_name == 'vgg':
                layer = model.features[25]
            elif model_name == 'squeezenet':
                layer = model.features[12]
            elif model_name == 'densenet':
                layer = model.features.denseblock4.denselayer16
            elif model_name == 'inception':
                layer = model.Mixed_7c
        explainer = GuidedGradCam(model, layer)
    else:
        explainers = ['lime', 'integratedGradient', 'gradientShap', 'kernelShap', 'occlusion', 'guidedBackProp',
                      'deepLift', 'guidedGradCAM']
        raise ValueError(f'Explainer not supported. Use {explainers}')

    return explainer


def get_image_attributions(explainer, inputImg, target=None, method=None, randomSeed=888, **kwargs):
    """ Returns attribution scores for the specified image and explainer. extra kwargs are sent to the .attribute
        of each explainer. The input image is assumed to be of shape (c x h x w) """

    if method == 'integratedGradient':
        attributions = explainer.attribute(inputImg, target=target, n_steps=200, **kwargs)
    elif method == 'gradientShap':
        torch.manual_seed(randomSeed)
        np.random.seed(randomSeed)
        attributions = explainer.attribute(inputImg,
                                           n_samples=50,
                                           stdevs=0.0001,
                                           baselines=torch.zeros(inputImg.shape),
                                           target=target,
                                           **kwargs)
    elif method == 'kernelShap':
        # todo not working, why is the reshape needed here but not in the others?
        attributions = explainer.attribute(inputImg.reshape(1, 3, 32, 32),
                                           baselines=0,
                                           target=target,
                                           **kwargs)
    elif method == 'occlusion':
        assert 'sliding_window_shapes' in kwargs, 'the sliding_window_shapes param. should be specified according' \
                                                  'to the documentation of captum.attr.Occlusion.attribute'
        attributions = explainer.attribute(inputImg,
                                           strides=1,
                                           target=target,
                                           baselines=0,
                                           **kwargs)
    elif method == 'lime':
        # todo why is the reshape needed here but not in the others? Why does it always return 0?
        attributions = explainer.attribute(inputImg.reshape(1, 3, 32, 32), target=target, **kwargs)
    elif method in {'guidedBackProp', 'deepLift', 'guidedGradCAM'}:
        attributions = explainer.attribute(inputImg, target=target, **kwargs)
    else:
        explainers = ['lime', 'integratedGradient', 'gradientShap', 'kernelShap', 'occlusion', 'guidedBackProp',
                      'deepLift', 'guidedGradCAM']
        raise ValueError(f'Explainer not supported. Use {explainers}')

    return attributions


def lime_image_attributions(model, data, labelsToExplain):
    rbf_kernel = get_exp_kernel_similarity_function('euclidean', kernel_width=0.75 * sqrt(len(data[0])))

    limeAttr = LimeBase(model,
                        SkLearnLinearModel("linear_model.Ridge"),
                        similarity_func=rbf_kernel,
                        perturb_func=_lime_perturb_func,
                        perturb_interpretable_space=False,
                        from_interp_rep_transform=None,
                        to_interp_rep_transform=_lime_identity)
    if len(data.shape) != 4:
        raise ValueError('Data should be 4D (nObs x channels x height x width).')

    attr = torch.FloatTensor([])
    for obs, label in zip(data, labelsToExplain):
        attr = torch.cat((attr, limeAttr.attribute(obs.reshape(1, 3, 32, 32))))

    return attr.detach().numpy()


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


def torch_pixel_attributions(model, data, labelsToExplain, method='lime', randomState=888, **kwargs):
    """ Get model attributions as feature importance vectors for torch models working on tabular data. Note that the
    input images must be alredy be able to be forwarded to the model without any further transformations.

    :param model: (object) trained Pytorch model.
    :param data: (Tensor) data for which to get attributions.
    :param labelsToExplain: (Tensor) class labels the attributions will be computed w.r.t.
    :param method: (str) attribution method: 'lime', 'shap'.
    :param randomState: (int) random seed.
    :param kwargs: extra keyword arguments for the .attribute method of the explainer.
    :return: (ndarray) normalized [0, 1] observation feature attributions
    """

    torch.manual_seed(randomState)

    explainer = get_image_explainer(model, method=method)

    attributions = []
    for image, target in zip(data, labelsToExplain):
        attr = get_image_attributions(explainer, inputImg=image, target=target, method=method, **kwargs)
        attr = attr.squeeze()
        if len(attr.shape) == 3:
            attr = attr.squeeze().reshape(attr.shape[1], attr.shape[2], attr.shape[0])
            # mean pool channel attributions
            attr = attr.mean(dim=2)
        elif len(attr.shape) != 2:
            raise ValueError(f'Attribution shape {attr.shape} is not valid.')

        attributions.append(normalize_array(attr.detach().numpy()))

    return attributions
