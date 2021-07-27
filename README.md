# teex

[![PyPI Version](https://img.shields.io/pypi/v/teex)](https://img.shields.io/pypi/v/teex)
[![Open GitHub Issues](https://img.shields.io/github/issues/chus-chus/teex)](https://img.shields.io/github/issues/chus-chus/teex)
[![Documentation Status](https://readthedocs.org/projects/teex/badge/?version=latest)](https://teex.readthedocs.io/en/latest/?badge=latest)

A Python **T**oolbox for the **E**valuation of machine learning **Ex**planations.

This project aims to provide a simple way of evaluating individual black box explanations. Moreover, it contains a collection
of easy-to-access datasets with available ground truth explanations.

## Installation

The teex package is on [PyPI](https://pypi.org/project/teex/). To install it, simply run

```shell
pip install teex
```

teex is compatible with Python >= 3.6.

## Usage overview

teex is divided into subpackages, one for each explanation type. Each subpackage contains two modules:

- **eval**: contains evaluation methods for that particular explanation type. For every subpackage, there is one high-level
  functions to easily compute all the available metrics.
- **data**: contains data classes with available g.t. explanations of that particular 
            explanation type, both synthetic and real. All of them are objects that need to be instanced and, when sliced,
            will return the data, the target and the ground truth explanations, respectively.
  
### Feature Importance

In **teex**, feature importance vectors are a universal representation: we can 'translate' all other explanation types
to feature importance vectors.

**What are feature importance vectors?** They are vectors with one entry per feature. Each entry contains a weight that 
represents a feature's importance for the observation's outcome. Weights are usually in the range [-1, 1]. 


<p align="center">
    <img src="https://raw.githubusercontent.com/slundberg/shap/master/docs/artwork/boston_instance.png" 
         alt="drawing" width="650"/>

Fig. 1 <a href="https://github.com/slundberg/shap">SHAP</a> values, each representing impact on model output of 
each feature.
</p>

Popular feature importance model-agnostic explainers are, for example, [SHAP](https://github.com/slundberg/shap) or 
[LIME](https://github.com/marcotcr/lime). Because weights in each method represent slighlty different things, 
we make the assumption that they all mean roughly the same if they are in the same range (if we want to compare methods).
**teex** performs this mapping automatically if necessary.

```python 
from teex.featureImportance.data import SenecaFI
from teex.featureImportance.eval import feature_importance_scores
from sklearn import DecisionTreeClassifier
import shap

# generate artificial data
X, y, exps = SenecaFI(nSamples=500, nFeatures=5)[:]

# instance and train the model
model = DecisionTreeClassifier()
model.fit(X, y)

# predict individual feature importance explanations
explainer = shap.Explainer(model)
predictedSHAPs = explainer(X).values

# evaluate predicted explanations againsy ground truths (implicit range mappings)
feature_importance_scores(exps, predictedSHAPs, metrics=['fscore', 'cs', 'rec'])
```

### Saliency Maps

A saliency map is an image that shows each pixel's unique quality. In our context, each pixel (feature) contains a score (in the
ranges [-1, 1] or [0, 1] as before) that represents a likelihood or probability of belonging to a particular class. 
For example:

<p align="center">
    <img src="https://www.mdpi.com/entropy/entropy-22-01365/article_deploy/html/images/entropy-22-01365-g001.png" 
         alt="drawing" width="350"/>

Fig. 2 Input image and saliency map for the prediction of the class "dog" overlayed on top of the original image. It tells us where the model "looks" when issuing the prediction. <a href="https://www.mdpi.com/1099-4300/22/12/1365">source</a>
</p>

**teex** contains artificial and real-life datasets for saliency map explanations:

<p align="center">
    <img src="https://raw.githubusercontent.com/chus-chus/teex/master/docs/images/kahikatea.png" 
         alt="drawing" width="450"/>

Fig. 3 The <a href="https://zenodo.org/record/5059769#.YN7KKegzZPZ">Kahikatea</a> dataset. Contains aerial images with the task of identifying whether there are 
Kahikatea trees (a species endemic to New Zealand) in the area or not. Observation on the left, ground truth 
explanation on the right.
</p>

<p align="center">
    <img src="https://raw.githubusercontent.com/chus-chus/teex/master/docs/images/artificial_sm.png" 
         alt="drawing" width="400"/>

Fig. 4 Artificial image dataset with g.t. saliency map explanations.
</p>

A basic usage example of **teex** with saliency maps simulating perfect predicted explanations:

```python
from teex.saliencyMap.data import Kahikatea
from teex.saliencyMap.eval import saliency_map_scores

X, y, exps = Kahikatea()[:]
saliency_map_scores(exps[y == 1], exps[y == 1], metrics=['fscore', 'auc'])
```


### Decision Rules

WIP!

### Word Importance

WIP!

## Tutorials and API

The full API documentation can be found on [Read The Docs](https://teex.readthedocs.io).

Here are some sample notebooks with examples:

- [Generating image data with g.t. saliency map explanations](https://github.com/chus-chus/teex/blob/main/docs/demos/gen_saliency_maps_seneca.ipynb)

## Contributing

Before contributing to teex, please take a moment to read the [manual](https://github.com/chus-chus/teex/blob/main/CONTRIBUTING.md).

## Acknowledgements
This work has been made possible by the [University of Waikato](https://www.waikato.ac.nz/) under the scope of 
the [TAIAO](https://taiao.ai/) project.

<p align="center">
    <a href="https://taiao.ai">
        <img src="https://taiao.ai/img/Untitled.png" alt="drawing" width="50"/>
    </a>
    <a href="https://www.waikato.ac.nz/">
        <img src="https://upload.wikimedia.org/wikipedia/en/thumb/b/bd/University_of_Waikato_logo.svg/1200px-University_of_Waikato_logo.svg.png" alt="drawing" width="45"/>
    </a> 
    <a href="https://www.upc.edu/en">
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/97/Logo_UPC.svg/2048px-Logo_UPC.svg.png" alt="drawing" width="50"/>
    </a>
</p>

