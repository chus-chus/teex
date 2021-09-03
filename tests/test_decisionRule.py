import unittest

import numpy as np

from teex.decisionRule.data import Statement, DecisionRule, SenecaDR
from teex.decisionRule.eval import rule_scores, _AVAILABLE_DECISION_RULE_METRICS


class TestDecisionRuleClass(unittest.TestCase):
    """ Tests for the DecisionRule class. """

    def setUp(self) -> None:
        self.c1 = Statement('a', lowB=2, upperB=3)  # 2 < a < 3

    def test_rule_def(self):
        _ = DecisionRule([self.c1])

    def test_rule_def_insertion(self):
        r = DecisionRule([self.c1])
        c2 = Statement('b', lowB=3, upperB=5)
        r.insert_statement(c2)

    def test_rule_def_upsert(self):
        r = DecisionRule([self.c1])
        c2 = Statement('a', lowB=3, upperB=5)
        r.upsert_statement(c2)


class TestDRDataSeneca(unittest.TestCase):
    """ Tests for the DR seneca artificial data generation class. """

    def setUp(self) -> None:
        self.nSamples = 5
        self.nFeatures = 3
        self.data = SenecaDR(self.nSamples, self.nFeatures)
        self.X, self.y, self.exps = self.data[:]

    def test_seneca_types(self):
        self.assertIsInstance(self.X, np.ndarray)
        self.assertIsInstance(self.y, np.ndarray)
        self.assertIsInstance(self.exps, list)
        for exp in self.exps:
            self.assertIsInstance(exp, DecisionRule)

    def test_seneca_shapes(self):
        self.assertEqual(self.X.shape, (self.nSamples, self.nFeatures))
        self.assertEqual(self.y.shape, (self.nSamples,))
        self.assertEqual(len(self.exps), self.nSamples)

    def test_seneca_feature_representation(self):
        """ Checks than in 20 samples all features appear in some explanations. """
        self.nSamples = 20
        self.nFeatures = 3
        self.data = SenecaDR(self.nSamples, self.nFeatures)
        self.X, self.y, self.exps = self.data[:]
        allFeatures = self.data.featureNames
        usedFeatures = {feat: False for feat in allFeatures}
        for exp in self.exps:
            for feature in exp.get_features():
                usedFeatures[feature] = True
        self.assertTrue(sum(usedFeatures.values()) == len(allFeatures))


class TestDRMetrics(unittest.TestCase):
    """ Tests for the decisionRule metrics. """

    def setUp(self) -> None:
        self.metrics = list(_AVAILABLE_DECISION_RULE_METRICS)

    def test_right(self):
        c1 = Statement('a', lowB=2, upperB=3, lowOp='<=', upperOp='<=')  # 2 < a < 3
        scores = rule_scores(DecisionRule([c1]), DecisionRule([c1]), metrics=self.metrics,
                             allFeatures=['a', 'b'])
        self.assertTrue((scores == np.ones(len(self.metrics))).all())

    def test_wrong(self):
        c1 = Statement('a', lowB=2, upperB=3, lowOp='<=', upperOp='<=')  # 2 < a < 3
        c2 = Statement('b', lowB=2, upperB=3, lowOp='<=', upperOp='<=')  # 2 < a < 3
        scores = rule_scores(DecisionRule([c1]), DecisionRule([c2]), metrics=self.metrics,
                             allFeatures=['a', 'b'])
        self.assertTrue((scores == np.zeros(len(self.metrics))).all())

    def test_edge_cases(self):
        c1 = Statement('x', lowB=2, upperB=3, lowOp='<=', upperOp='<=')  # 2 < x < 3
        _ = rule_scores(DecisionRule([c1]), DecisionRule([c1]), metrics=self.metrics,
                        allFeatures=['a', 'b'])  # will crash if metrics not defined


if __name__ == '__main__':
    unittest.main()
