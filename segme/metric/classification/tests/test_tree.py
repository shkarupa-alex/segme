import numpy as np
import tensorflow as tf
from keras.src import backend, testing

from segme.metric.classification.tree import HeinsenTreeAccuracy


class TestHeinsenTreeAccuracy(testing.TestCase):
    def test_config(self):
        metric = HeinsenTreeAccuracy(TREE_PATHS, name="metric1")
        self.assertEqual(metric.name, "metric1")

    def test_false(self):
        metric = HeinsenTreeAccuracy(TREE_PATHS)
        metric.update_state(TREE_TARGETS, 1.0 - TREE_LOGITS)
        self.assertAlmostEqual(metric.result(), 0.0, decimal=7)

    def test_true(self):
        metric = HeinsenTreeAccuracy(TREE_PATHS)
        metric.update_state(TREE_TARGETS, TREE_LOGITS)
        self.assertAlmostEqual(metric.result(), 1.0, decimal=7)

    def test_value_level3(self):
        logits = TREE_LOGITS.copy()
        logits[0, 5] = 4.0

        metric = HeinsenTreeAccuracy(TREE_PATHS)
        metric.update_state(TREE_TARGETS, logits)
        self.assertAlmostEqual(metric.result(), 0.9166667, decimal=7)

    def test_value_level2(self):
        logits = TREE_LOGITS.copy()
        logits[0, 2] = 4.0

        metric = HeinsenTreeAccuracy(TREE_PATHS)
        metric.update_state(TREE_TARGETS, logits)
        self.assertAlmostEqual(metric.result(), 0.8333334, decimal=7)

    def test_value_level1(self):
        logits = TREE_LOGITS.copy()
        logits[0, 1] = 4.0

        metric = HeinsenTreeAccuracy(TREE_PATHS)
        metric.update_state(TREE_TARGETS, logits)
        self.assertAlmostEqual(metric.result(), 0.75, decimal=7)

    def test_weight_level2(self):
        logits = TREE_LOGITS.copy()
        logits[0, 2] = 4.0

        weights = np.array([[0.5], [1.0], [1.0], [1.0]], "float32")

        metric = HeinsenTreeAccuracy(TREE_PATHS)
        metric.update_state(TREE_TARGETS, logits, weights)
        self.assertAlmostEqual(metric.result(), 0.9047619, decimal=7)

    def test_batch(self):
        logits = np.random.uniform(size=TREE_LOGITS.shape)

        metric = HeinsenTreeAccuracy(TREE_PATHS)
        for i in range(logits.shape[0]):
            metric.update_state(TREE_TARGETS[i : i + 1], logits[i : i + 1])
        res0 = backend.convert_to_numpy(metric.result())

        metric.reset_state()
        metric.update_state(TREE_TARGETS, logits)
        res1 = metric.result()

        self.assertEqual(res0, res1)


TREE_PATHS = [[0, 3, 4], [0], [1], [0, 2], [0, 3], [0, 3, 5]]
TREE_LOGITS = np.array(
    [
        [1.134725, -1.853591, -1.793681, 0.253602, 2.649923, -0.321642],
        [-0.545508, 0.588591, 0.235042, 0.300561, -0.918901, -0.152296],
        [0.837255, -1.03309, -0.190284, 0.837352, 0.257713, 1.214522],
        [0.813726, -1.356368, 1.323276, -0.316505, 0.125477, -1.459085],
    ],
    "float32",
)
TREE_TARGETS = np.array([[4], [1], [5], [2]], "int32")
