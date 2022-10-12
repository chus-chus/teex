import unittest
import numpy as np

from teex.decisionRule.data import Statement, DecisionRule, SenecaDR, clean_binary_statement, \
    rule_to_feature_importance, _induce_binary_statement, \
    _generate_binary_statement, str_to_decision_rule, \
    rulefit_to_decision_rule
from teex.decisionRule.eval import rule_scores, _AVAILABLE_DECISION_RULE_METRICS

class TestStatement(unittest.TestCase):
    
    def test_absence_bound(self):
        self.assertRaises(ValueError, Statement, 'a', '<', 2)
        
    def test_lowerb_higher_upperb(self):
        self.assertRaises(ValueError, Statement, 'a', '<', 5, '>', 6)
        
    def test_eq(self):
        s = Statement('a', 1.5)
        self.assertEqual(s, s)
        
    def test_eq_bin(self):
        s = Statement('a',lowOp='<',lowB=2,upperOp='<',upperB=5)
        self.assertEqual(s, s)
        
    def test_neq(self):
        s = Statement('a',lowOp='<',lowB=2,upperOp='<',upperB=5)
        a = Statement('b',lowOp='<',lowB=2,upperOp='<',upperB=5)
        self.assertNotEqual(s, a)
        
    def test_str(self):
        s = Statement('a',lowOp='<',lowB=2,upperOp='<',upperB=5)
        self.assertEqual(s.__str__, s.__str__)
        
    def test_wrong_op(self):
        self.assertRaises(ValueError, Statement, 'a', 1.5, 'Invalid!')
        
    def test_wrong_op_bin(self):
        self.assertRaises(ValueError, Statement, 'a', 'Invalid_1', 5, 'Invalid_2', 6)


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
        
    def test_len(self):
        self.assertEqual(len(self.data), 5)

class TestDRDataUtils(unittest.TestCase):
    
    def setUp(self) -> None:
        self.nSamples = 2
        self.nFeatures = 2
        self.data = SenecaDR(self.nSamples, self.nFeatures)
        self.X, self.y, self.exps = self.data[:]
    
    def test_r_fi(self):
        r = rule_to_feature_importance(self.exps, self.data.featureNames)
        self.assertTrue((r == np.array([[0., 1.], [0., 1.]])).all())
        
    def test_induce_statements(self):
        self.assertIsNotNone(_induce_binary_statement('a', '<', 3))
        self.assertIsNotNone(_induce_binary_statement('a', '>', 3))
        self.assertRaises(ValueError, _induce_binary_statement, 'a', 'Invalid', 3)
        
    def test_induce_binary_statements(self):
        self.assertIsNotNone(_generate_binary_statement('a', '<', 3, '>', 1))
        self.assertIsNotNone(_generate_binary_statement('a', '>', 1, '<', 3))
        self.assertRaises(ValueError, _generate_binary_statement, 'a', 'Invalid!', 1, '<', 3)
        
    def test_clean_binary_statements(self):
        r = clean_binary_statement([('>', 1), ('<', 1)])
        self.assertEqual(r, ('>', 1, '<', 1))
        self.assertRaises(ValueError, clean_binary_statement, [('Invalid', 1)])
        
    def test_str_to_dr(self):
        r = 'a != 2.5 -> res > 3'
        self.assertEqual("IF 'a' != 2.5 THEN 'res' > 3.0", 
                         str_to_decision_rule(r,'unary').__str__())
        r = 'a <= 2.5 & a > 1 -> res > 3'
        self.assertEqual("IF 1.0 < 'a' <= 2.5 THEN 'res' > 3.0", 
                         str_to_decision_rule(r,'binary').__str__())
        r = 'a <= 2.5 & a > 1 & b > 1 -> res > 3 & res <= 5'
        self.assertEqual("IF 1.0 < 'a' <= 2.5, 1.0 < 'b' THEN 3.0 < 'res' <= 5.0", 
                         str_to_decision_rule(r,'binary').__str__())
        r = 'a <= 2.5 & a > 1 & b > 1 -> res = class0'        
        self.assertEqual("IF 1.0 < 'a' <= 2.5, 1.0 < 'b' THEN 'res' = class0", 
                         str_to_decision_rule(r,'binary').__str__())
        r = 'a <= 2.5 & a > 1 & a > 2 -> res = class0'        
        self.assertEqual("IF 2.0 < 'a' <= 2.5 THEN 'res' = class0", 
                         str_to_decision_rule(r,'binary').__str__())
        self.assertRaises(ValueError, str_to_decision_rule, "a < 1 -> res = class", "invalid!")
        r = 'a <= 2.5'        
        self.assertEqual("IF 'a' <= 2.5 THEN None", 
                         str_to_decision_rule(r,'binary').__str__())
        
    def test_rulefit_to_dr(self):
        
        import pandas as pd
        from rulefit import RuleFit
        
        boston_data = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv')
        y = boston_data.medv.values
        features = boston_data.columns
        X = boston_data.drop("medv", axis=1).values
        rf = RuleFit(random_state=888)
        rf.fit(X[:10], y[:10], feature_names=features)
        dRules, _ = rulefit_to_decision_rule(rf.get_rules())
        self.assertTrue(isinstance(dRules[0], DecisionRule))
        

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
