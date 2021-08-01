import unittest
from teex.wordImportance.data import Newsgroup
from teex.wordImportance.eval import _AVAILABLE_WORD_IMPORTANCE_METRICS, word_importance_scores
import numpy as np


class TestWIDataNewsgroup(unittest.TestCase):
    """ Tests for the WI Newsgroup class. """

    def setUp(self) -> None:
        self.x, self.y, self.e = Newsgroup()[:]

    def test_newsgroup_types(self):
        self.assertIsInstance(self.x, list)
        self.assertIsInstance(self.y, list)
        self.assertIsInstance(self.e, list)
        for exp in self.e:
            self.assertIsInstance(exp, dict)

    def test_newsgroup_shapes(self):
        self.assertEqual(len(self.x), len(self.y), len(self.x))
        for exp in self.e:
            self.assertTrue(exp != {})


class TestWIMetrics(unittest.TestCase):
    """ Tests for the wordImportance metrics. """

    def setUp(self) -> None:
        self.metrics = list(_AVAILABLE_WORD_IMPORTANCE_METRICS)

    def test_right(self):
        w = {'test': 1, 'this': .5}
        scores = word_importance_scores(w, w, metrics=self.metrics)
        self.assertTrue((scores == np.ones(len(self.metrics))).all())

        w = [{'test': 1, 'this': .5}, {'test': 1, 'this': .5}]
        scores = word_importance_scores(w, w, metrics=self.metrics)
        self.assertTrue((scores == np.ones(len(self.metrics))).all())

        scores = word_importance_scores(w, w, metrics=self.metrics, average=False)
        self.assertTrue((scores == np.ones((2, len(self.metrics)))).all())

    def test_wrong(self):
        w1 = {'test': 1, 'this': 0}
        w2 = {'test': 0, 'this': 1}
        scores = word_importance_scores(w1, w2, metrics=self.metrics)
        self.assertTrue((scores == np.zeros(len(self.metrics))).all())

        w1 = [{'test': 1, 'this': 0} for _ in range(3)]
        w2 = [{'test': 0, 'this': 1} for _ in range(3)]
        scores = word_importance_scores(w1, w2, metrics=self.metrics)
        self.assertTrue((scores == np.zeros(len(self.metrics))).all())

        scores = word_importance_scores(w1, w2, metrics=self.metrics, average=False)
        self.assertTrue((scores == np.zeros((3, len(self.metrics)))).all())


if __name__ == '__main__':
    unittest.main()
