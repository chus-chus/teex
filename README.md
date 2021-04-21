# TAIAOexp

A python package for explanation experimentation. It provides methods for:

* Generation of synthetic data with available ground truth explanations:

|**Type of synthetic data**|**Type of g.t. explanations**|
|----------------|-----------------------------|
|Tabular [1]     |Feature Importance           |
|Tabular [1]     |Decision Rules               |
|Images [1]      |Pixel Importance             |

* Transparent models able to provide ground truth explanations. They provide feature importance, decision rules and 
  pixel importance explanations.  
  

* Explanation quality metrics. Given a generated explanation and a ground-truth, multiple scores are provided:

|**Type of g.t. explanations**|**Quality Metric**       |
|-----------------------------|-------------------------|
|Feature Importance, Decision Rules, Pixel Importance / SM               |Cosine Similarity [1]    |
|Feature Importance, Decision Rules, Pixel Importance / SM               |Precision [1]            |
|Feature Importance, Decision Rules, Pixel Importance / SM               |Recall [1]               |
|Feature Importance, Decision Rules, Pixel Importance / SM               |Fß-Score [1]             |
|Feature Importance, Decision Rules, Pixel Importance / SM               |AUC (ROC)                |
|Decision Rules                                                          |Complete Rule Quality [1]|

* Utility functions

    * image_utils.binary_mask_rgb_image: create a binary mask from an RBG image mask.

[1] [Evaluating local explanation methods on ground truth](https://www.researchgate.net/publication/346916247_Evaluating_local_explanation_methods_on_ground_truth)
