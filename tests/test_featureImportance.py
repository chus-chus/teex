import unittest

import numpy as np
import os

from teex.featureImportance.data import SenecaFI
from teex.featureImportance.eval import _AVAILABLE_FEATURE_IMPORTANCE_METRICS, feature_importance_scores


class TestFIDataSeneca(unittest.TestCase):
    """ Test for feature importance data generation with seneca """

    def setUp(self) -> None:
        self.nSamples = 5
        self.nFeatures = 2
        self.senecaFI = SenecaFI(nSamples=self.nSamples, 
                                 nFeatures=self.nFeatures, 
                                 randomState=888)
        self.model = self.senecaFI.transparentModel
        self.X, self.y, self.exps = self.senecaFI[:]

    def test_seneca_types(self):
        self.assertIsInstance(self.X, np.ndarray)
        self.assertIsInstance(self.y, np.ndarray)
        self.assertIsInstance(self.exps, np.ndarray)

    def test_seneca_shapes(self):
        self.assertEqual(self.X.shape, (self.nSamples, self.nFeatures))
        self.assertEqual(self.y.shape, (self.nSamples,))
        self.assertEqual(self.exps.shape, (self.nSamples, self.nFeatures))

    def test_seneca_exp_bounds(self):
        mins = np.min(self.exps)
        maxs = np.max(self.exps)
        self.assertTrue(mins >= -1)
        self.assertTrue(maxs <= 1)
        
    def test_seneca_model_predict_proba(self):
        arr = np.array([[ 0.2, -0.1],
                        [ 0.5, -0.9],
                        [-0.6,  0.5]])
        r = self.model.predict_proba(arr)
        exp = np.array([[0.3, 0.6],
                        [0.7, 0.2],
                        [0.6, 0.3]])
        self.assertTrue((np.trunc(r * 10) == np.trunc(exp * 10)).all())


class TestFIMetrics(unittest.TestCase):
    """ Test for feature importance metrics """

    def setUp(self) -> None:
        self.metrics = list(_AVAILABLE_FEATURE_IMPORTANCE_METRICS)

    def test_metrics_right(self):
        gts = np.array([[1, 0], [0, 1]])
        preds = np.array([[1, 0], [0, 1]])
        res = feature_importance_scores(gts, preds, metrics=self.metrics, average=False)
        self.assertTrue((res == np.ones(len(self.metrics))).all())

    def test_metrics_wrong(self):
        gts = np.array([[0, 1], [1, 0]])
        preds = np.array([[1, 0], [0, 1]])
        res = feature_importance_scores(gts, preds, metrics=self.metrics, average=False)
        self.assertTrue((res == np.zeros(len(self.metrics))).all())

    def test_edge_cases(self):
        gts = np.array([0, 0, 0])
        preds = np.array([0, 0, 0])
        _ = feature_importance_scores(gts, preds, metrics=self.metrics)  # will crash if metrics not defined
