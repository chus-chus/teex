# TAIAOexp

A python package for explanation experimentation. It provides:

* A way to generate synthetic data, with available ground truth explanations. The data will be of different kinds:

|**Type of data**|**Type of g.t. explanations**|
|Tabular [1]     |Feature Importance           |
|Tabular [1]     |Decision Rules               |
|Images [1]      |Pixel Importance             |

* Explanation quality metrics. Given a generated explanation and a ground-truth, multiple scores are provided:

|**Type of g.t. explanations**|**Quality Metric**       |
|Feature Importance           |Cosine Similarity [1]    |
|Decision Rules               |Precision [1]            |
|Decision Rules               |Recall [1]               |
|Decision Rules               |Fß-Score [1]             |
|Decision Rules               |Complete Rule Quality [1]|
|Pixel Importance             |Precision                |
|Pixel Importance             |Recall                   |
|Pixel Importance             |Fß-Score                 |

[1] [Evaluating local explanation methods on ground truth](evernote:///view/223543586/s738/b3946cf1-7da3-dd59-d2f2-db83c12a6301/8979951d-2c39-198b-bea9-3e9d23159cba/)
