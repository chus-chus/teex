# TAIAOexp

A python package for explanation experimentation. It provides:

* Transparent models with available ground truth explanations. The data used to train them is synthetically generated
  and of different kinds:

|**Type of data**|**Type of g.t. explanations**|
|----------------|-----------------------------|
|Tabular [1]     |Feature Importance           |
|Tabular [1]     |Decision Rules               |
|Images [1]      |Pixel Importance             |

* Explanation quality metrics. Given a generated explanation and a ground-truth, multiple scores are provided:

|**Type of g.t. explanations**|**Quality Metric**       |
|-----------------------------|-------------------------|
|Feature Importance           |Cosine Similarity [1]    |
|Decision Rules               |Precision [1]            |
|Decision Rules               |Recall [1]               |
|Decision Rules               |Fß-Score [1]             |
|Decision Rules               |Complete Rule Quality [1]|
|Pixel Importance / SM        |Precision                |
|Pixel Importance / SM        |Recall                   |
|Pixel Importance / SM        |Fß-Score                 |
|Saliency maps (SM)           |AUC (ROC)                |

* Utility functions

    * image_utils.binary_mask_rgb_image: create a binary mask from an RBG image mask.

[1] [Evaluating local explanation methods on ground truth](https://www.researchgate.net/publication/346916247_Evaluating_local_explanation_methods_on_ground_truth)
