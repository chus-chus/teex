# teex: a toolbox for evaluating XAI explanations

[![PyPI Version](https://img.shields.io/pypi/v/teex)](https://img.shields.io/pypi/v/teex)
[![Open GitHub Issues](https://img.shields.io/github/issues/chus-chus/teex)](https://img.shields.io/github/issues/chus-chus/teex)
[![Documentation Status](https://readthedocs.org/projects/teex/badge/?version=latest)](https://teex.readthedocs.io/en/latest/?badge=latest)

A Python **T**oolbox for the **E**valuation of machine learning **Ex**planations.

This project aims to provide a simple way of **evaluating** individual black box explanations against ground truth. Moreover, it contains a collection of easy-to-access datasets with available g.t. explanations.

## Installation

The teex package is on [PyPI](https://pypi.org/project/teex/). To install it, simply run

```shell
pip install teex
```

**teex** is compatible with Python >= 3.6.

## Usage overview

teex is divided into subpackages, one for each explanation type. Each subpackage contains two modules, focused on two
distinct functionalities:

- **eval**: contains _**evaluation**_ methods for that particular explanation type. For every subpackage, there is one high-level
  functions to easily compute all the available metrics.
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

downloads and assigns data from the Kahikatea [1] dataset:

<p align="center">
    <img src="https://raw.githubusercontent.com/chus-chus/teex/master/docs/images/kahikatea.png" 
         alt="drawing" width="450"/>

Fig. 1 The <a href="https://zenodo.org/record/5059769#.YN7KKegzZPZ">Kahikatea</a> dataset. Contains aerial images with the task of identifying whether there are 
Kahikatea trees (a species endemic to New Zealand) in the area or not. Observation on the left, ground truth 
explanation on the right.
</p>

Synthetic datasets can also be generated effortlessly:

```python
from teex.saliencyMap.data import SenecaSM

X, y, exps = SenecaSM()[:]
```

<p align="center">
    <img src="https://raw.githubusercontent.com/chus-chus/teex/master/docs/images/artificial_sm.png" 
         alt="drawing" width="400"/>

Fig. 2 Artificial image dataset with g.t. saliency map explanations.
</p>

## Tutorials and API

The full API documentation can be found on [Read The Docs](https://teex.readthedocs.io).

Here are some sample notebooks with examples.

---
*Saliency maps*
- [Retrieving image data with g.t. saliency map explanations](https://github.com/chus-chus/teex/blob/main/docs/demos/gen_saliency_map.ipynb)
- [Evaluating Captum saliency map explanations](https://github.com/chus-chus/teex/blob/main/docs/demos/eval_saliency_map.ipynb)
---
*Feature importance vectors*
- [Retrieving tabular data with g.t. feature importance explanations](https://github.com/chus-chus/teex/blob/main/docs/demos/gen_feature_importance.ipynb)
- [Evaluating LIME feature importance explanations](https://github.com/chus-chus/teex/blob/main/docs/demos/eval_feature_importance.ipynb)
---
*Decision rules*
- [Retrieving tabular data with g.t. decision rule explanations](https://github.com/chus-chus/teex/blob/main/docs/demos/gen_decision_rule.ipynb)
- [Evaluating decision rule explanations](https://github.com/chus-chus/teex/blob/main/docs/demos/eval_decision_rule.ipynb)
---
*Word importance vectors*
- [Retrieving language data with g.t. word importance explanations](https://github.com/chus-chus/teex/blob/main/docs/demos/gen_word_importance.ipynb)
- [Evaluating word importance explanations](https://github.com/chus-chus/teex/blob/main/docs/demos/eval_word_importance.ipynb)


## Contributing

There is still work to do. Before contributing to **teex**, please take a moment to read the [manual](https://github.com/chus-chus/teex/blob/main/CONTRIBUTING.md).

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

## References

[1] Y. Jia et al. (2021) 
Studying and Exploiting the Relationship Between Model Accuracy and Explanation Quality, 
ECML-PKDD 2021

