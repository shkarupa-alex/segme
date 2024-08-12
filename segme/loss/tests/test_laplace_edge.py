import numpy as np
import tensorflow as tf
from keras.src import layers
from keras.src import models
from keras.src import testing

from segme.loss.laplace_edge import LaplaceEdgeCrossEntropy
from segme.loss.laplace_edge import laplace_edge_cross_entropy
from segme.loss.tests.test_common_loss import BINARY_LOGITS
from segme.loss.tests.test_common_loss import BINARY_TARGETS
from segme.loss.tests.test_common_loss import BINARY_WEIGHTS
from segme.loss.tests.test_common_loss import MULTI_LOGITS
from segme.loss.tests.test_common_loss import MULTI_TARGETS


class TestLaplaceEdgeCrossEntropy(testing.TestCase):
    def test_config(self):
        loss = LaplaceEdgeCrossEntropy(reduction="none", name="loss1")
        self.assertEqual(loss.name, "loss1")
        self.assertEqual(loss.reduction, "none")

    def test_zeros(self):
        logits = -10.0 * tf.ones((3, 16, 16, 1), "float32")
        targets = tf.zeros((3, 16, 16, 1), "int32")

        result = laplace_edge_cross_entropy(
            y_true=targets,
            y_pred=logits,
            sample_weight=None,
            from_logits=True,
            force_binary=False,
        )

        self.assertAllClose(result, [0.0] * 3, atol=1e-4)

    def test_ones(self):
        logits = 10.0 * tf.ones((3, 16, 16, 1), "float32")
        targets = tf.ones((3, 16, 16, 1), "int32")

        result = laplace_edge_cross_entropy(
            y_true=targets,
            y_pred=logits,
            sample_weight=None,
            from_logits=True,
            force_binary=False,
        )

        self.assertAllClose(result, [0.0] * 3, atol=1e-4)

    def test_false(self):
        logits = -10.0 * tf.ones((3, 16, 16, 1), "float32")
        targets = tf.ones((3, 16, 16, 1), "int32")

        result = laplace_edge_cross_entropy(
            y_true=targets,
            y_pred=logits,
            sample_weight=None,
            from_logits=True,
            force_binary=False,
        )

        self.assertAllClose(result, [0.0] * 3, atol=1e-4)

    def test_true(self):
        logits = 10.0 * tf.ones((3, 16, 16, 1), "float32")
        targets = tf.zeros((3, 16, 16, 1), "int32")

        result = laplace_edge_cross_entropy(
            y_true=targets,
            y_pred=logits,
            sample_weight=None,
            from_logits=True,
            force_binary=False,
        )

        self.assertAllClose(result, [0.0] * 3, atol=1e-4)

    def test_value(self):
        loss = LaplaceEdgeCrossEntropy(from_logits=True)
        result = loss(BINARY_TARGETS, BINARY_LOGITS)

        # self.assertAlmostEqual(result, 3.879105925) # for zero padding and 1-channel BCE
        self.assertAlmostEqual(result, 4.2626038, decimal=4)

    def test_weight(self):
        loss = LaplaceEdgeCrossEntropy(from_logits=True)

        result = loss(BINARY_TARGETS[:, :, :2], BINARY_LOGITS[:, :, :2])
        self.assertAlmostEqual(result, 2.9807048, decimal=4)

        result = loss(BINARY_TARGETS, BINARY_LOGITS, BINARY_WEIGHTS)
        self.assertAlmostEqual(result, 3.1198683, decimal=4)

        result = loss(BINARY_TARGETS, BINARY_LOGITS, BINARY_WEIGHTS * 2.0)
        self.assertAlmostEqual(result, 3.1198683 * 2, decimal=4)

    def test_multi(self):
        loss = LaplaceEdgeCrossEntropy(from_logits=True)
        result = loss(MULTI_TARGETS, MULTI_LOGITS)
        self.assertAlmostEqual(result, 3.5500488, decimal=4)

    def test_batch(self):
        probs = np.random.rand(2, 224, 224, 1).astype("float32")
        targets = (np.random.rand(2, 224, 224, 1) > 0.5).astype("int32")

        loss = LaplaceEdgeCrossEntropy(from_logits=True)
        result0 = loss(targets, probs)
        result1 = (
            sum([loss(targets[i : i + 1], probs[i : i + 1]) for i in range(2)])
            / 2
        )

        self.assertAlmostEqual(result0, result1, decimal=5)

    def test_model(self):
        model = models.Sequential([layers.Dense(5, activation="sigmoid")])
        model.compile(
            loss="SegMe>Loss>LaplaceEdgeCrossEntropy",

        )
        model.fit(np.zeros((2, 16, 16, 1)), np.zeros((2, 16, 16, 1), "int32"))
        models.Sequential.from_config(model.get_config())
