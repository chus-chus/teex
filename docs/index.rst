.. teex documentation master file, created by
   sphinx-quickstart on Tue Jun 29 12:41:10 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: images/teex_logo__.png
   :width: 115
   :align: center
   :alt: Our AI generated logo. Comes from the prompt: 
      'logo of a t, inspired by an AI that is fair and responsible.

================================
teex
================================

A Python Toolbox for the evaluation of machine learning explanations.

This project aims to provide a simple way of evaluating individual black box 
explanations. Moreover, it contains a collection of easy-to-access datasets with available ground truth explanations.

Visit our `GitHub <https://github.com/chus-chus/teex>`_ for source.

--------------------------
1. Usage overview
--------------------------

``teex`` is divided into subpackages, one for each explanation type. Each subpackage contains two modules, focused on two
distinct functionalities:

- ``eval``: contains *evaluation* methods for that particular explanation type. For every subpackage, there is one high-level
  function to easily compute all the available metrics for an arbitrary number of explanations.
- ``data``: contains *data* classes with available g.t. explanations of that particular 
  explanation type, both synthetic and real. All of them are objects that need to be instanced and, when sliced,
  will return the data, the target and the ground truth explanations, respectively.


1.1. Evaluation (with feature importance as an example)
=======================================================

**What are feature importance vectors?** They are vectors with one entry per feature. Each entry contains a weight that 
represents a feature's importance for the observation's outcome. Weights are usually in the range :math:`[-1, 1]`.

Suppose that we have a dataset with available g.t. explanations (``gtExps``) and a model trained with it (``model``):

.. code-block:: python

   from teex.featureImportance.eval import feature_importance_scores

   # get individual feature importance explanations with any method
   predictedExps = get_explanations(model, X)

   # evaluate predicted explanations against ground truths
   feature_importance_scores(gtExps, predictedExps, metrics=['fscore', 'cs', 'auc'])


This basic syntax is followed by the main evaluation APIs of all 4 explanation types:

- **Feature Importance**: ``feature_importance_scores``
- **Saliency Maps**: ``saliency_map_scores``
- **Decision Rules**: ``rule_scores``
- **Word Importance**: ``word_importance_scores``

Other functionalities are included in each evaluation module.

1.2. Metrics supported
========================

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
  
  - **Complete Rule Quality**: Proportion of lower and upper bounds in a rule explanation whose that are :math:`\epsilon`-close to the respective lower and upper bounds (same feature) in the ground truth rule explanation amongst those that are not infinity.
  - All metrics in feature importance, where a transformation of the rule into feature importance vectors is performed first. See doc. for details.
- **Word Importance**:
  
  - All metrics in feature importance, where a vocabulary is considered the feature space and a word importance explanation may or may not contain words from the vocabulary.

Note how in **teex**, feature importance vectors are a universal representation: we 'translate' all other explanation types
to feature importance vectors to allow a wider metric space.  

1.3. Datasets
===============

**teex** also provides an easy way to get and use data with available ground truth explanations. It contains real datasets and can generate synthetic ones.
All of them are instanced as objects, and can be sliced as usual. For example:

.. code-block:: python

   from teex.saliencyMap.data import Kahikatea
   X, y, exps = Kahikatea()[:]


downloads and assigns data from the Kahikatea dataset:

.. raw:: html

   <p style = "text-align: center;">
      <img src="https://raw.githubusercontent.com/chus-chus/teex/master/docs/images/kahikatea_sample.png" 
            alt="drawing" width="200"/>
      <img src="https://raw.githubusercontent.com/chus-chus/teex/master/docs/images/kahikatea_gt.png" alt="drawing" width="200"/>
   </p>
   <body>
   <p style = "text-align: center;">Fig. 1 A <a href="https://zenodo.org/record/5059769#.YN7KKegzZPZ">Kahikatea</a> dataset sample. </p>
   </body>

Other datasets, such as `CUB-200-2011 <https://www.vision.caltech.edu/datasets/cub_200_2011/>`_ and 
the `Oxford-IIIT Pet Dataset <https://www.robots.ox.ac.uk/~vgg/data/pets/>`_, are available on **teex**, with over 19000 images 
and 230 distinct classes combined:

.. code-block:: python

   from teex.saliencyMap.data import CUB200
   X, y, exps = CUB200()[:]

.. raw:: html

   <p style = "text-align: center;">
      <img src="https://raw.githubusercontent.com/chus-chus/teex/master/docs/images/cub_sample.jpg" 
            alt="drawing" width="200"/>
      <img src="https://raw.githubusercontent.com/chus-chus/teex/master/docs/images/cub_gt.png" alt="drawing" width="200"/>
   </p>
   <body>
   <p style = "text-align: center;">Fig. 2 A <a href="https://www.vision.caltech.edu/datasets/cub_200_2011/">CUB-200-2011</a> dataset sample.  </p>
   </body>

.. code-block:: python

   from teex.saliencyMap.data import OxfordIIIT
   X, y, exps = OxfordIIIT()[:]

.. raw:: html
      
   <p style = "text-align: center;">
      <img src="https://raw.githubusercontent.com/chus-chus/teex/master/docs/images/ox_sample.jpg" 
            alt="drawing" width="200"/>
      <img src="https://raw.githubusercontent.com/chus-chus/teex/master/docs/images/ox_gt.png" alt="drawing" width="200"/>
   </p>
   <body>
   <p style = "text-align: center;">Fig. 3 An <a href="https://www.robots.ox.ac.uk/~vgg/data/pets/">Oxford-IIIT Pet Dataset</a> sample. </p>
   </body>


Synthetic datasets can also be easily generated:

.. code-block:: python

   from teex.saliencyMap.data import SenecaSM
   X, y, exps = SenecaSM()[:]


.. raw:: html
   
   <p style = "text-align: center;">
      <img src="https://raw.githubusercontent.com/chus-chus/teex/master/docs/images/seneca_sm_sample.png" 
            alt="drawing" width="200"/>
      <img src="https://raw.githubusercontent.com/chus-chus/teex/master/docs/images/seneca_sm_gt.png" alt="drawing" width="200"/>
   </p>
   <body>
   <p style = "text-align: center;">Fig. 4 Artificial image and its g.t. saliency map explanation.
   </p>
   </body>

Datasets for all other explanation types are available too.

----------------
2. Examples
----------------

Here you will find hands-on examples of ``teex``. For each explanation type, explanation evaluation and data generation are showcased. 
Advanced examples and experiments can also be found within the ``saliency map`` and ``model selection`` sections.

.. toctree:: 
   :maxdepth: 2

   demos/examples

----------------
3. API reference
----------------

.. toctree::
   :maxdepth: 1

   api/modules
