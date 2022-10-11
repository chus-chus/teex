import unittest

import numpy as np

from teex.saliencyMap.data import CUB200, OxfordIIIT, \
    SenecaSM, Kahikatea, binarize_rgb_mask, delete_sm_data
from teex.saliencyMap.eval import saliency_map_scores, \
    _AVAILABLE_SALIENCY_MAP_METRICS
from teex._datasets.info.kahikatea import _kahikateaNEntries
from teex._datasets.info.OxfordIIT_Pet import _oxford_iit_length


class TestSMDataSenecea(unittest.TestCase):
    """ Test for saliency map data generation with seneca """

    def setUp(self) -> None:
        self.nSamples = 4
        randomState = 7
        self.imageH, self.imageW = 32, 32
        self.patternH, self.patternW = 16, 16
        cellHeight, cellWidth = 4, 4
        self.patternProp = 0.5
        fillPct = 0.4
        colorDev = 0.5

        self.data = SenecaSM(nSamples=self.nSamples, imageH=self.imageH, imageW=self.imageW, patternH=self.patternH,
                             patternW=self.patternW, cellH=cellHeight, cellW=cellWidth, patternProp=self.patternProp,
                             fillPct=fillPct, colorDev=colorDev, randomState=randomState)
        
        self.model = self.data.transparentModel

        self.X, self.y, self.exps = self.data[:]

    def test_seneca_types(self):
        self.assertIsInstance(self.X, np.ndarray)
        self.assertIsInstance(self.y, np.ndarray)
        self.assertIsInstance(self.exps, np.ndarray)

    def test_seneca_shapes(self):
        self.assertEqual(self.X.shape, (self.nSamples, self.imageH, self.imageW, 3))
        self.assertEqual(self.y.shape, (self.nSamples,))
        self.assertEqual(self.exps.shape, (self.nSamples, self.imageH, self.imageW))
        self.assertEqual(self.data.pattern.shape, (self.patternH, self.patternW, 3))

    def test_seneca_label_proportion(self):
        self.assertEqual(np.sum(self.y)/len(self.y), self.patternProp)
        
    def test_seneca_len(self):
        self.assertEqual(len(self.data), self.nSamples)
        
    def test_model_predict(self):
        self.assertTrue((self.model.predict(self.X[0]) == self.y[0]).all())
        
    def test_model_predict_proba(self):
        r = self.model.predict_proba(self.X[:2])
        trueRes = np.array([[1., 0.],
                            [0., 1.]])
        self.assertTrue((r == trueRes).all())
        
    def test_model_explain(self):
        r = self.model.explain(self.X[:2])
        self.assertTrue((r == self.exps[:2]).all())
        
    def test_model_has_pat(self):
        r1, _ = self.model._has_pattern(self.X[0], retIndices=True)
        r2 = self.model._has_pattern(self.X[0], retIndices=False)
        
        self.assertEqual(r1, r2)
              

class TestSMCUB200(unittest.TestCase):
    """ Test for the CUB-200-2011 dataset """

    def setUp(self) -> None:
        self.data = CUB200()

    def test_slices(self):
        d = self.data[:11]
        self.assertIsNotNone(d[:10])
        self.assertIsNotNone(d[1:10])
        self.assertIsNotNone(d[:10:2])
        for i, t in enumerate(d):
            if i != 1:
                for e in t:
                    e.close()

    def test_class_loading(self):
        for c in self.data.classMap.keys():
            obs = self.data.get_class_observations(c)
            self.assertIsNotNone(obs)
            # close pointers, as they generate alloc warnings in -Wall calls
            for i, t in enumerate(obs):
                if i != 1:
                    for e in t:
                        e.close()
                        
    def test_wrong_class(self):
        self.assertRaises(ValueError, self.data.get_class_observations, None)
                        
    def test_slice(self):
        self.assertIsNotNone(self.data[0])
        
    def test_slice_invalid(self):
        self.assertRaises(TypeError, self.data.__getitem__, "invalid slice!")
                        

class TestSMOxfordIIIT(unittest.TestCase):
    """ Test for the Oxford IIIT Pet Dataset """

    def setUp(self) -> None:
        self.data = OxfordIIIT()

    def test_slices(self):
        d = self.data[:11]
        self.assertIsNotNone(d[:10])
        self.assertIsNotNone(d[1:10])
        self.assertIsNotNone(d[:10:2])
        for i, t in enumerate(d):
            if i != 1:
                for e in t:
                    e.close()

    def test_class_loading(self):
        for c in self.data.classMap.keys():
            obs = self.data.get_class_observations(c)
            self.assertIsNotNone(obs)
            # close pointers, as they generate alloc warnings in -Wall calls
            for i, t in enumerate(obs):
                if i != 1:
                    for e in t:
                        e.close()
                        
    def test_wrong_class(self):
        self.assertRaises(ValueError, self.data.get_class_observations, None)
                        
    def test_slice(self):
        self.assertIsNotNone(self.data[0])
        
    def test_slice_invalid(self):
        self.assertRaises(TypeError, self.data.__getitem__, "invalid slice!")
        
    def test_len(self):
        self.assertEqual(len(self.data), _oxford_iit_length)


class TestSMKahikatea(unittest.TestCase):
    """ Test for saliency map data generation with seneca """

    def setUp(self) -> None:
        self.data = Kahikatea()

    def test_slices(self):
        self.assertIsNotNone(self.data[:10])
        self.assertIsNotNone(self.data[1:10])
        self.assertIsNotNone(self.data[:10:2])
        
    def test_wrong_slice(self):
        self.assertRaises(TypeError, self.data.__getitem__, None)
        
    def test_get_class(self):
        self.assertIsNotNone(self.data.get_class_observations(0))
        
    def test_len(self):
        self.assertEqual(len(self.data), _kahikateaNEntries)
        
        
class TestSMMetrics(unittest.TestCase):

    def setUp(self) -> None:
        self.metrics = _AVAILABLE_SALIENCY_MAP_METRICS

    def test_metrics_real_rgb_image(self):
        _, _, mask = Kahikatea()[0]
        mask = np.array(mask)
        # perturb ground truth mask
        pred = binarize_rgb_mask(mask, bgValue='high')
        scores = saliency_map_scores(mask, pred, metrics=self.metrics)
        self.assertFalse(np.isnan(scores).any())

    def test_metrics_real_grayscale_image(self):
        _, _, mask = Kahikatea()[0]
        mask = np.array(mask)
        # perturb ground truth mask
        mask = binarize_rgb_mask(mask, bgValue='high')
        pred = mask
        scores = saliency_map_scores(mask, pred, metrics=self.metrics)
        self.assertTrue((scores == np.ones(len(self.metrics))).all())

    def test_metrics_right(self):
        gt = np.array([[[1, 0], [1, 0], [1, 0]],
                       [[1, 0], [1, 0], [1, 0]]])
        pred = gt
        scores = saliency_map_scores(gt, pred, metrics=self.metrics, average=False)
        self.assertTrue((scores == np.ones(len(self.metrics))).all())

    def test_n_outputs(self):
        gt = np.array([[[1, 0], [1, 0], [1, 0]],
                       [[1, 0], [1, 0], [1, 0]]])
        pred = gt
        scores = saliency_map_scores(gt, pred, metrics=self.metrics, average=False)
        self.assertEqual(len(scores), len(gt))

    def test_metrics_wrong(self):
        gt = np.array([[[1, 0], [1, 0], [1, 0]],
                       [[1, 0], [1, 0], [1, 0]]])
        pred = np.array([[[0, 1], [0, 1], [0, 1]],
                         [[0, 1], [0, 1], [0, 1]]])
        scores = saliency_map_scores(gt, pred, metrics=self.metrics, average=False)
        self.assertTrue((scores == np.zeros(len(self.metrics))).all())
        

class TestPublicDataUtils(unittest.TestCase):
    
    def test_delete_sm_data(self):
        _ = Kahikatea()
        delete_sm_data()
        self.assertRaises(FileNotFoundError, delete_sm_data)
        
    def test_bin_rgb(self):
        self.assertRaises(ValueError, binarize_rgb_mask, np.array([]), "invalid!")
