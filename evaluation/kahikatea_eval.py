import numpy as np
import torch
from PIL import Image
from torchvision.transforms import transforms

from evaluation.image import saliency_map_scores
from explanation.images import torch_pixel_attributions
from utils.image import list_images, binarize_rgb_mask

if __name__ == '__main__':
    def transform_img_kahikatea(image):
        transformKahikatea = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        transform_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        image = Image.fromarray(np.uint8(image)).convert('RGB')

        transformedImg = transformKahikatea(image)

        inputImg = transform_normalize(transformedImg).unsqueeze(0)
        return inputImg, transformedImg

    # read kahikatea
    pathName = '/Users/chusantonanzas/GDrive/U/4t/TFG/TAIAOexp_CLI/data/Kahikatea/data/ve_positive'
    posKahikatea = list_images(pathName, returnType='list')

    pathName = '/Users/chusantonanzas/GDrive/U/4t/TFG/TAIAOexp_CLI/data/Kahikatea/expl/ve_positive'
    explanations = list_images(pathName, returnType='list')
    # binarize ground truth masks:
    explanations = [binarize_rgb_mask(exp, bgColor='light') for exp in explanations]

    # load model
    modelPath = '/Users/chusantonanzas/GDrive/U/4t/TFG/TAIAOexp_CLI/torch_models/Kahikatea/alexnet.pth'
    model = torch.load(modelPath, map_location='cpu')

    # prediction labels and transform to tensors individually
    targets = torch.LongTensor([1 for _ in range(len(posKahikatea))])

    nImagesToEval = 1

    # apply image transformations for input to the models
    print('transforming images...')
    posKahikateaTransformed = []
    for im in posKahikatea[:nImagesToEval]:
        inputImg, _ = transform_img_kahikatea(im)
        posKahikateaTransformed.append(torch.FloatTensor(inputImg))

    # evaluate the model explanations for the prediction of positive class
    print('computing attributions...')
    attrs = torch_pixel_attributions(model, posKahikateaTransformed, targets[:nImagesToEval], method='integratedGradient')

    # reshape the gt explanations so that they have the same shape as the explanations (which belong to transformed)
    # input images.
    transformedExps = []
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    for exp in explanations[:nImagesToEval]:
        transformedExp = transform(Image.fromarray(np.uint8(exp))).ceil().detach().numpy()
        transformedExps.append(transformedExp)

    # and evaluate the attributions
    metrics = ['auc', 'fscore', 'prec', 'rec', 'cs']
    scores = []
    for i, att in enumerate(attrs):
        scores.append(saliency_map_scores(transformedExps[i], att, metrics=metrics, binarizeGt=False))

    scores = np.mean(np.array(scores), axis=0)

    for i, metric in enumerate(metrics):
        print(f'Mean {metric}: {scores[i]}')

