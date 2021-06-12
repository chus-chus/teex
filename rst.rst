https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html

.. code::

    code
    block!

.. math::

        q(e, \tilde e) = \frac{1}{N_{\not\infty}} \sum_{i=1}^{|e|}{\delta_{\varepsilon}(e_i, \tilde e_i)}, \text{where}

        \delta_{\varepsilon}(e_i, \tilde e_i) = \begin{cases}
                                                    1 \text{if } |e_i - \tilde{e}_i| \leq \varepsilon \wedge |e_i| \neq
                                                        \infty \wedge |\tilde{e}_i| \neq \infty, \\
                                                    0 \text{otherwise}
                                                \end{cases}

Generate synthetic classification tabular data with ground truth decision rule explanations. The returned
decision rule explanations are instances of the :code:`DecisionRule` class, generated with the
:code:`TransparentRuleClassifier` class.

:param nSamples: (int) number of samples to be generated.
:param nFeatures: (int) total number of features in the generated data.
:param returnModel: (bool) should the :code:`TransparentRuleClassifier` model used for the generation of the explanations be returned?
:param featureNames: (array-like) names of the generated features.
:param randomState: (int) random seed.
:return: (ndarrays) X, y, explanations, featureNames (if they were not specified),
                    model (optional, param 'returnModel').

:class:`rule.TransparentRuleClassifier`

>>> asdfa
2 2

fjdasf

1. a
2. b
    3. a

:param val: Value for the statement (if not binary)

1. a
2. b
3. c
    3.1 a

a ver las mates :math:`\frac{ \sum_{t=0}^{N}f(t,k) }{N}` jajaj


A conjunction of statements as conditions that imply a result. Internally, the rule is represented as a
dictionary of statements with the feature names as keys. A feature cannot have more than one Statement (
Statements can be binary). This class is capable of adapting previous Statements depending on new Statements that
are added to it with the upsert method (see :code:`.upsert_condition` method).

>>> 1 + 1
2

Generate synthetic classification tabular data with ground truth decision rule explanations. The returned
decision rule explanations are instances of the :code:`DecisionRule` class, generated with the
:class:`rule.TransparentRuleClassifier` class.

:param nSamples: (int) number of samples to be generated.
:param nFeatures: (int) total number of features in the generated data.
:param returnModel: (bool) should the :code:`rule.TransparentRuleClassifier` model used for the generation of the explanations be returned?
:param featureNames: (array-like) names of the generated features.
:param randomState: (int) random state seed.
:return:
    - X (ndarray) of shape (n_obs, n_features) Generated data
    - y (ndarray) of shape (n_obs,) Ground truth explanations
    - featureNames (list) list with the generated feature names (if they were not specified)
    - model (:class:`rule.TransparentRuleClassifier`) Model instance used to generate the data (returned if :code:`returnModel` is True)

