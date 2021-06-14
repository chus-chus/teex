import numpy as np
import torch
from captum.attr import Lime, IntegratedGradients, GradientShap, KernelShap, Occlusion, GuidedBackprop, DeepLift, GuidedGradCam


def get_explainer(model, layer=None, method='guidedGradCAM', model_name=None, **kwargs):
    """ Gets a captum explainer based on the provided PyTorch model.
    :param model: (callable) PyTorch classification model
    :param layer: (object) Model layer to use guidedGradCAM on. Predefined layer numbers are automatically defined if
    model_name is in ['resnet18', 'resnet50', 'alexnet', 'vgg', 'squeezenet', 'densenet', 'inception'].
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


def get_attributions(explainer, obs, target=None, method=None, randomSeed=888, **kwargs):
    """ Returns attribution scores for an observation and explainer. The shape of the observation,
        be it image or tabular, must be accepted by the model which the explainer was initialized on top of.
        Extra kwargs are sent to the .attribute of each Captum explainer."""

    if method == 'integratedGradient':
        attributions = explainer.attribute(obs, target=target, n_steps=200, **kwargs)
    elif method == 'gradientShap':
        torch.manual_seed(randomSeed)
        np.random.seed(randomSeed)
        attributions = explainer.attribute(obs,
                                           n_samples=50,
                                           stdevs=0.0001,
                                           baselines=torch.zeros(obs.shape),
                                           target=target,
                                           **kwargs)
    elif method == 'kernelShap':
        # todo reshape should not be needed.
        # todo not working, why is the reshape needed here but not in the others?
        attributions = explainer.attribute(obs.reshape(1, 3, 32, 32),
                                           baselines=0,
                                           target=target,
                                           **kwargs)
    elif method == 'occlusion':
        assert 'sliding_window_shapes' in kwargs, 'the sliding_window_shapes param. should be specified according' \
                                                  'to the documentation of captum.attr.Occlusion.attribute'
        attributions = explainer.attribute(obs,
                                           strides=1,
                                           target=target,
                                           baselines=0,
                                           **kwargs)
    elif method == 'lime':
        # todo why is the reshape needed here but not in the others? Why does it always return 0?
        attributions = explainer.attribute(obs.reshape(1, 3, 32, 32), target=target, **kwargs)
    elif method in {'guidedBackProp', 'deepLift', 'guidedGradCAM'}:
        attributions = explainer.attribute(obs, target=target, **kwargs)
    else:
        explainers = ['lime', 'integratedGradient', 'gradientShap', 'kernelShap', 'occlusion', 'guidedBackProp',
                      'deepLift', 'guidedGradCAM']
        raise ValueError(f'Explainer not supported. Use {explainers}')

    return attributions
