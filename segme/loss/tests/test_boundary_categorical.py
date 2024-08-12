import numpy as np
import tensorflow as tf
from keras.src import layers
from keras.src import models
from keras.src import testing

from segme.loss.boundary_categorical import BoundaryCategoricalLoss
from segme.loss.boundary_categorical import boundary_categorical_loss
from segme.loss.tests.test_common_loss import BINARY_LOGITS
from segme.loss.tests.test_common_loss import BINARY_TARGETS
from segme.loss.tests.test_common_loss import BINARY_WEIGHTS
from segme.loss.tests.test_common_loss import MULTI_LOGITS
from segme.loss.tests.test_common_loss import MULTI_TARGETS


class TestBoundaryCategoricalLoss(testing.TestCase):
    def test_config(self):
        loss = BoundaryCategoricalLoss(reduction="none", name="loss1")
        self.assertEqual(loss.name, "loss1")
        self.assertEqual(loss.reduction, "none")

    def test_zeros(self):
        logits = -10.0 * tf.ones((3, 16, 16, 1), "float32")
        targets = tf.zeros((3, 16, 16, 1), "int32")

        result = boundary_categorical_loss(
            y_true=targets, y_pred=logits, sample_weight=None, from_logits=True
        )

        self.assertAllClose(result, [0.0] * 3, atol=1e-4)

    def test_ones(self):
        logits = 10.0 * tf.ones((3, 16, 16, 1), "float32")
        targets = tf.ones((3, 16, 16, 1), "int32")

        result = boundary_categorical_loss(
            y_true=targets, y_pred=logits, sample_weight=None, from_logits=True
        )

        self.assertAllClose(result, [0.0] * 3, atol=1e-4)

    def test_false(self):
        logits = -10.0 * tf.ones((3, 16, 16, 1), "float32")
        targets = tf.ones((3, 16, 16, 1), "int32")

        result = boundary_categorical_loss(
            y_true=targets, y_pred=logits, sample_weight=None, from_logits=True
        )

        self.assertAllClose(result, [0.0] * 3, atol=1e-4)

    def test_true(self):
        logits = 10.0 * tf.ones((3, 16, 16, 1), "float32")
        targets = tf.zeros((3, 16, 16, 1), "int32")

        result = boundary_categorical_loss(
            y_true=targets, y_pred=logits, sample_weight=None, from_logits=True
        )

        self.assertAllClose(result, [0.0] * 3, atol=1e-4)

    def test_value(self):
        loss = BoundaryCategoricalLoss(from_logits=True)
        result = loss(BINARY_TARGETS, BINARY_LOGITS)

        self.assertAlmostEqual(result, 0.18070064)

    def test_weight(self):
        loss = BoundaryCategoricalLoss(from_logits=True)

        result = loss(BINARY_TARGETS[:, :, :2], BINARY_LOGITS[:, :, :2])
        self.assertAlmostEqual(result, 0.1696591)

        result = loss(BINARY_TARGETS, BINARY_LOGITS, BINARY_WEIGHTS)
        self.assertAlmostEqual(result, 0.15673386)

        result = loss(BINARY_TARGETS, BINARY_LOGITS, BINARY_WEIGHTS * 2.0)
        self.assertAlmostEqual(result, 0.15673386 * 2.0, decimal=6)

    def test_multi(self):
        loss = BoundaryCategoricalLoss(from_logits=True)
        result = loss(MULTI_TARGETS, MULTI_LOGITS)

        self.assertAlmostEqual(result, 0.34384584)

    def test_batch(self):
        probs = np.random.rand(2, 224, 224, 1).astype("float32")
        targets = (np.random.rand(2, 224, 224, 1) > 0.5).astype("int32")

        loss = BoundaryCategoricalLoss(from_logits=True)
        result0 = loss(targets, probs)
        result1 = (
            sum([loss(targets[i : i + 1], probs[i : i + 1]) for i in range(2)])
            / 2
        )

        self.assertAlmostEqual(result0, result1, decimal=6)

    def test_model(self):
        model = models.Sequential([layers.Dense(5, activation="sigmoid")])
        model.compile(
            loss="SegMe>Loss>BoundaryCategoricalLoss",
            
        )
        model.fit(np.zeros((2, 16, 16, 1)), np.zeros((2, 16, 16, 1), "int32"))
        models.Sequential.from_config(model.get_config())
