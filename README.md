<p align="center">
    <img src="https://raw.githubusercontent.com/chus-chus/teex/master/docs/images/teex_logo__.png" 
         alt="Our AI generated logo. Comes from the prompt: 'logo of a t, inspired by an AI that is fair and responsible.'" width="115"/>

# teex: a toolbox for evaluating XAI explanations

[![PyPI Version](https://img.shields.io/pypi/v/teex)](https://img.shields.io/pypi/v/teex)
[![Open GitHub Issues](https://img.shields.io/github/issues/chus-chus/teex)](https://img.shields.io/github/issues/chus-chus/teex)
[![codecov](https://codecov.io/gh/chus-chus/teex/branch/main/graph/badge.svg?token=PWSRR5NZTQ)](https://codecov.io/gh/chus-chus/teex)
![Build Status](https://github.com/chus-chus/teex/workflows/CI/badge.svg?branch=main)
[![Documentation Status](https://readthedocs.org/projects/teex/badge/?version=latest)](https://teex.readthedocs.io/en/latest/?badge=latest)

A Python **t**oolbox for the **e**valuation of machine learning **ex**planations.

This project aims to provide a simple way of **evaluating** individual black box explanations against ground truth. Moreover, it contains a collection of easy-to-access datasets with available g.t. explanations.

## Installation

The teex package is on [PyPI](https://pypi.org/project/teex/). To install it, simply run

```shell
pip install teex
```

**teex** is compatible with Python 3.8 and 3.9.

## Documentation

**teex**'s documentation and API reference can be found on [Read The Docs](https://teex.readthedocs.io).

## Usage overview

teex is divided into subpackages, one for each explanation type. Each subpackage contains two modules, focused on two
distinct functionalities:

- **eval**: contains _**evaluation**_ methods for that particular explanation type. For every subpackage, there is one high-level
  function to easily compute all the available metrics for an arbitrary number of explanations.
- **data**: contains _**data**_ classes with available g.t. explanations of that particular 
            explanation type, both synthetic and real. All of them are objects that need to be instanced and, when sliced,
            will return the data, the target and the ground truth explanations, respectively.
  
### Evaluation (with feature importance as an example)

**What are feature importance vectors?** They are vectors with one entry per feature. Each entry contains a weight that 
represents a feature's importance for the observation's outcome. Weights are usually in the range [-1, 1].

Suppose that we have a dataset with available g.t. explanations (``gtExps``) and a model trained with it (``model``):

```python
from teex.featureImportance.eval import feature_importance_scores

# get individual feature importance explanations with any method
predictedExps = get_explanations(model, X)

# evaluate predicted explanations against ground truths
feature_importance_scores(gtExps, predictedExps, metrics=['fscore', 'cs', 'auc'])
```

This basic syntax is followed by the main evaluation APIs of all 4 explanation types:

- **Feature Importance**: ``feature_importance_scores``
- **Saliency Maps**: ``saliency_map_scores``
- **Decision Rules**: ``rule_scores``
- **Word Importance**: ``word_importance_scores``

Other functionalities are included in each evaluation module. More about each explanation type can be found in the example notebooks and the documentation.

#### Metrics supported:

Metrics available as of v1.0.0 are

- **Feature Importance**
  - **Cosine Similarity**: similarity between the two vectors is measured in an inner product space in terms of orientation.
  - **ROC AUC**: where the ground truth is binarized in order for it to represent a class and the predicted vector entries are interpreted as classification scores or likelihood.
  - **F1 Score**: where both ground truth and prediction are binarized according to a user-defined threshold.
  - **Precision**: g.t. and prediction treated as in F1 Score
  - **Recall**: g.t. and prediction treated as in F1 Score
- **Saliency Maps**
  - Same metrics as in feature importance. Each pixel in an image is considered to be a feature.
- **Decision Rules**
  - **Complete Rule Quality**: Proportion of lower and upper bounds in a rule explanation whose that are eps-close to the respective lower and upper bounds (same feature) in the ground truth rule explanation amongst those that are not infinity.
  - All metrics in feature importance, where a transformation of the rule into feature importance vectors is performed first. See doc. for details.
- **Word Importance**:
  - All metrics in feature importance, where a vocabulary is considered the feature space and a word importance explanation may or may not contain words from the vocabulary.

Note how in **teex**, feature importance vectors are a universal representation: we 'translate' all other explanation types
to feature importance vectors to allow a wider metric space.  


### Data

**teex** also provides an easy way to get and use data with available ground truth explanations. It contains real datasets and can generate synthetic ones.
All of them are instanced as objects, and can be sliced as usual. For example:

```python
from teex.saliencyMap.data import Kahikatea

X, y, exps = Kahikatea()[:]
```

downloads and assigns data from the Kahikatea dataset:    

<p align="center">
    <img src="https://raw.githubusercontent.com/chus-chus/teex/master/docs/images/kahikatea_sample.png" 
         alt="drawing" width="200"/>
    <img src="https://raw.githubusercontent.com/chus-chus/teex/master/docs/images/kahikatea_gt.png" alt="drawing" width="200"/>
</p>
<body>
  <p align="center">Fig. 1 A <a href="https://zenodo.org/record/5059769#.YN7KKegzZPZ">Kahikatea</a> dataset sample. </p>
</body>

Other datasets, such as [CUB-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/) and the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/), are available on **teex**, with over 19000 images and 230 distinct classes:

<p align="center">
    <img src="https://raw.githubusercontent.com/chus-chus/teex/master/docs/images/cub_sample.jpg" 
         alt="drawing" width="200"/>
    <img src="https://raw.githubusercontent.com/chus-chus/teex/master/docs/images/cub_gt.png" alt="drawing" width="200"/>
</p>
<body>
  <p align="center">Fig. 2 A <a href="https://www.vision.caltech.edu/datasets/cub_200_2011/">CUB-200-2011</a> dataset sample.  </p>
</body>
    
<p align="center">
    <img src="https://raw.githubusercontent.com/chus-chus/teex/master/docs/images/ox_sample.jpg" 
         alt="drawing" width="200"/>
    <img src="https://raw.githubusercontent.com/chus-chus/teex/master/docs/images/ox_gt.png" alt="drawing" width="200"/>
</p>
<body>
  <p align="center">Fig. 3 An <a href="https://www.robots.ox.ac.uk/~vgg/data/pets/">Oxford-IIIT Pet Dataset</a> sample. </p>
</body>


Synthetic datasets can also be easily generated:

```python
from teex.saliencyMap.data import SenecaSM

X, y, exps = SenecaSM()[:]
```

<p align="center">
    <img src="https://raw.githubusercontent.com/chus-chus/teex/master/docs/images/seneca_sm_sample.png" 
         alt="drawing" width="200"/>
    <img src="https://raw.githubusercontent.com/chus-chus/teex/master/docs/images/seneca_sm_gt.png" alt="drawing" width="200"/>
</p>
<body>
  <p align="center">Fig. 4 Artificial image and its g.t. saliency map explanation.
 </p>
</body>


## Tutorials and demos

---
- [Improving model selection with explanation quality](https://teex.readthedocs.io/en/latest/demos/model_selection/model_selection_nb.html)
---
*Saliency maps*
- [Retrieving image data with g.t. saliency map explanations](https://teex.readthedocs.io/en/latest/demos/saliency_map/gen_saliency_map_nb.html)
- [Evaluating Captum saliency map explanations](https://teex.readthedocs.io/en/latest/demos/saliency_map/eval_saliency_map_nb.html)
---
*Feature importance vectors*
- [Retrieving tabular data with g.t. feature importance explanations](https://teex.readthedocs.io/en/latest/demos/feature_importance/gen_feature_importance_nb.html)
- [Evaluating LIME feature importance explanations](https://teex.readthedocs.io/en/latest/demos/feature_importance/eval_feature_importance_nb.html)
---
*Decision rules*
- [Retrieving tabular data with g.t. decision rule explanations](https://teex.readthedocs.io/en/latest/demos/decision_rule/gen_decision_rule_nb.html)
- [Evaluating decision rule explanations](https://teex.readthedocs.io/en/latest/demos/decision_rule/eval_decision_rule_nb.html)
---
*Word importance vectors*
- [Retrieving language data with g.t. word importance explanations](https://teex.readthedocs.io/en/latest/demos/word_importance/gen_word_importance_nb.html)
- [Evaluating word importance explanations](https://teex.readthedocs.io/en/latest/demos/word_importance/eval_word_importance_nb.html)


## Contributing

There is still work to do and we would really appreciate your help. Before contributing to **teex**, please take a moment to read the [manual](https://github.com/chus-chus/teex/blob/main/CONTRIBUTING.md).

## Acknowledgements
This work has been made possible by the [University of Waikato](https://www.waikato.ac.nz/) under the scope of 
the [TAIAO](https://taiao.ai/) project.

<p align="center">
    <a href="https://taiao.ai">
        <img src="https://taiao.ai/assets/TAIAO_logo.png" alt="drawing" width="150"/>
    </a>
    <a href="https://www.waikato.ac.nz/">
        <img src="https://upload.wikimedia.org/wikipedia/en/thumb/b/bd/University_of_Waikato_logo.svg/1200px-University_of_Waikato_logo.svg.png" alt="drawing" width="45"/>
    </a> 
    <a href="https://www.upc.edu/en">
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/97/Logo_UPC.svg/2048px-Logo_UPC.svg.png" alt="drawing" width="50"/>
    </a>
</p>
