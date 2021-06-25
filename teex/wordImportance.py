""" The word importance module. Contains all of the methods regarding this explanation type: synthetic data generation,
evaluation of explanations and related utils. """

# ===================================
#       EXPLANATION EVALUATION
# ===================================


def word_importance_scores(gts, preds, metrics=None, average=True):
    """ Quality metrics for word importance explanations, where each word is considered as a feature.

        :param gts: (dict, array-like of dicts) ground truth word importance/s, where each BOW is represented as a
            dictionary with words as keys and values as importances. Importances must be
        :param preds: (dict, array-like of dicts) predicted word importance/s, where each BOW is represented in the same
            way as the ground truths.
        :param metrics: (str / array-like of str, default=['auc']) Quality metric/s to compute. Available:

            - Classification scores:
                - 'auc': ROC AUC score. The value of each pixel of each saliency map in :code:`sMaps` is considered as a
                  prediction probability of the pixel pertaining to the salient class.
                - 'fscore': F1 Score.
                - 'prec': Precision Score.
                - 'rec': Recall score.

            - Similarity metrics:
                - 'cs': Cosine Similarity.

            For 'fscore', 'prec', 'rec' and 'cs', the saliency maps in :code:`sMaps` are binarized (see param
            :code:`binThreshold`).

        :param binThreshold: (float in [0, 1]) pixels of images in :code:`sMaps` with a value bigger than this will be set
            to 1 and 0 otherwise when binarizing for the computation of 'fscore', 'prec', 'rec' and 'auc'.
        :param gtBackgroundVals: (str) Only used when provided ground truth explanations are RGB. Color of the background
            of the g.t. masks 'low' if pixels in the mask representing the non-salient class are dark, 'high' otherwise).
        :param average: (bool, default :code:`True`) Used only if :code:`gts` and :code:`sMaps` contain multiple
            observations. Should the computed metrics be averaged across all of the samples?
        :return: (ndarray) specified metric/s in the original order. Can be of shape

            - (n_metrics,) if only one image has been provided in both :code:`gts` and :code:`sMaps` or when both are
              contain multiple observations and :code:`average=True`.
            - (n_metrics, n_samples) if :code:`gts` and :code:`sMaps` contain multiple observations and
              :code:`average=False`.

        """
