import unittest

import numpy as np

from teex.saliencyMap.data import CUB200, SenecaSM, Kahikatea, binarize_rgb_mask
from teex.saliencyMap.eval import saliency_map_scores, _AVAILABLE_SALIENCY_MAP_METRICS


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
        

class TestSMCUB200(unittest.TestCase):
    """ Test for saliency map data generation with seneca """

    def setUp(self) -> None:
        self.data = CUB200()

    def test_slices(self):
        d = self.data[:11]
        self.assertIsNotNone(d[:10])
        self.assertIsNotNone(d[1:10])
        self.assertIsNotNone(d[:10:2])

    def test_class_loading(self):
        for c in self.data.classMap.keys():
            obs = self.data.get_class_observations(c)
            self.assertIsNotNone(obs)
            
class TestSMKahikatea(unittest.TestCase):
    """ Test for saliency map data generation with seneca """

    def setUp(self) -> None:
        self.data = Kahikatea()

    def test_slices(self):
        self.assertIsNotNone(self.data[:10])
        self.assertIsNotNone(self.data[1:10])
        self.assertIsNotNone(self.data[:10:2])


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


if __name__ == '__main__':
    unittest.main()
