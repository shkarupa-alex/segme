import numpy as np
import tensorflow as tf
from keras.src import layers
from keras.src import models
from keras.src import testing

from segme.loss.mean_squared import MeanSquaredClassificationError
from segme.loss.mean_squared import MeanSquaredRegressionError
from segme.loss.mean_squared import mean_squared_classification_error
from segme.loss.mean_squared import mean_squared_regression_error
from segme.loss.tests.test_common_loss import BINARY_LOGITS
from segme.loss.tests.test_common_loss import BINARY_TARGETS
from segme.loss.tests.test_common_loss import BINARY_WEIGHTS
from segme.loss.tests.test_common_loss import MULTI_LOGITS
from segme.loss.tests.test_common_loss import MULTI_TARGETS


class TestMeanSquaredClassificationError(testing.TestCase):
    def test_config(self):
        loss = MeanSquaredClassificationError(reduction="none", name="loss1")
        self.assertEqual(loss.name, "loss1")
        self.assertEqual(loss.reduction, "none")

    def test_zeros(self):
        logits = -10.0 * tf.ones((3, 64, 64, 1), "float32")
        targets = tf.zeros((3, 64, 64, 1), "int32")

        result = mean_squared_classification_error(
            y_true=targets,
            y_pred=logits,
            sample_weight=None,
            from_logits=True,
            label_smoothing=0.0,
        )

        self.assertAllClose(result, [0.0] * 3, atol=6e-3)

    def test_ones(self):
        logits = 10 * tf.ones((3, 64, 64, 1), "float32")
        targets = tf.ones((3, 64, 64, 1), "int32")

        result = mean_squared_classification_error(
            y_true=targets,
            y_pred=logits,
            sample_weight=None,
            from_logits=True,
            label_smoothing=0.0,
        )

        self.assertAllClose(result, [0.0] * 3, atol=6e-3)

    def test_false(self):
        logits = -10.0 * tf.ones((3, 64, 64, 1), "float32")
        targets = tf.ones((3, 64, 64, 1), "int32")

        result = mean_squared_classification_error(
            y_true=targets,
            y_pred=logits,
            sample_weight=None,
            from_logits=True,
            label_smoothing=0.0,
        )

        self.assertAllClose(result, [1.0] * 3, atol=6e-3)

    def test_true(self):
        logits = 10.0 * tf.ones((3, 64, 64, 1), "float32")
        targets = tf.zeros((3, 64, 64, 1), "int32")

        result = mean_squared_classification_error(
            y_true=targets,
            y_pred=logits,
            sample_weight=None,
            from_logits=True,
            label_smoothing=0.0,
        )

        self.assertAllClose(result, [1.0] * 3, atol=6e-3)

    def test_value(self):
        logits = tf.tile(BINARY_LOGITS, [1, 16, 16, 1])
        targets = tf.tile(BINARY_TARGETS, [1, 16, 16, 1])

        loss = MeanSquaredClassificationError(from_logits=True)
        result = loss(targets, logits)

        self.assertAlmostEqual(result, 0.32667673, decimal=6)

    def test_weight(self):
        logits = tf.tile(BINARY_LOGITS, [1, 16, 16, 1])
        targets = tf.tile(BINARY_TARGETS, [1, 16, 16, 1])
        weights = tf.tile(BINARY_WEIGHTS, [1, 16, 16, 1])

        loss = MeanSquaredClassificationError(from_logits=True)

        result = loss(targets[:, :, :32], logits[:, :, :32])
        self.assertAlmostEqual(result, 0.32667673, decimal=6)

        result = loss(targets, logits, weights)
        self.assertAlmostEqual(result, 0.24974126, decimal=6)

        result = loss(targets, logits, weights * 2.0)
        self.assertAlmostEqual(result, 0.24974126 * 2, decimal=6)

    def test_multi(self):
        logits = tf.tile(MULTI_LOGITS, [1, 16, 16, 1])
        targets = tf.tile(MULTI_TARGETS, [1, 16, 16, 1])

        loss = MeanSquaredClassificationError(from_logits=True)
        result = loss(targets, logits)

        self.assertAlmostEqual(result, 0.2690816, decimal=6)

    def test_batch(self):
        probs = np.random.rand(2, 224, 224, 1).astype("float32")
        targets = (np.random.rand(2, 224, 224, 1) > 0.5).astype("int32")

        loss = MeanSquaredClassificationError(from_logits=True)
        result0 = loss(targets, probs)
        result1 = (
            sum([loss(targets[i : i + 1], probs[i : i + 1]) for i in range(2)])
            / 2
        )

        self.assertAlmostEqual(result0, result1, decimal=6)

    def test_model(self):
        model = models.Sequential([layers.Dense(5, activation="sigmoid")])
        model.compile(
            loss="SegMe>Loss>MeanSquaredClassificationError",
        )
        model.fit(np.zeros((2, 64, 64, 1)), np.zeros((2, 64, 64, 1), "int32"))
        models.Sequential.from_config(model.get_config())


class TestMeanSquaredRegressionError(testing.TestCase):
    def test_config(self):
        loss = MeanSquaredRegressionError(reduction="none", name="loss1")
        self.assertEqual(loss.name, "loss1")
        self.assertEqual(loss.reduction, "none")

    def test_zeros(self):
        logits = tf.zeros((3, 64, 64, 1), "float32")
        targets = tf.zeros((3, 64, 64, 1), "int32")

        result = mean_squared_regression_error(
            y_true=targets, y_pred=logits, sample_weight=None
        )

        self.assertAllClose(result, [0.0] * 3, atol=6e-3)

    def test_ones(self):
        logits = tf.ones((3, 64, 64, 1), "float32")
        targets = tf.ones((3, 64, 64, 1), "int32")

        result = mean_squared_regression_error(
            y_true=targets, y_pred=logits, sample_weight=None
        )

        self.assertAllClose(result, [0.0] * 3, atol=6e-3)

    def test_false(self):
        logits = tf.zeros((3, 64, 64, 1), "float32")
        targets = tf.ones((3, 64, 64, 1), "int32")

        result = mean_squared_regression_error(
            y_true=targets, y_pred=logits, sample_weight=None
        )

        self.assertAllClose(result, [1.0] * 3, atol=6e-3)

    def test_true(self):
        logits = tf.ones((3, 64, 64, 1), "float32")
        targets = tf.zeros((3, 64, 64, 1), "int32")

        result = mean_squared_regression_error(
            y_true=targets, y_pred=logits, sample_weight=None
        )

        self.assertAllClose(result, [1.0] * 3, atol=6e-3)

    def test_value(self):
        logits = tf.nn.sigmoid(BINARY_LOGITS)
        targets = tf.cast(BINARY_TARGETS, "float32")

        loss = MeanSquaredRegressionError()
        result = loss(targets, logits)

        self.assertAlmostEqual(result, 0.32667673, decimal=6)

    def test_weight(self):
        logits = tf.nn.sigmoid(BINARY_LOGITS)
        targets = tf.cast(BINARY_TARGETS, "float32")
        weights = BINARY_WEIGHTS

        loss = MeanSquaredRegressionError()

        result = loss(targets[:, :, :32], logits[:, :, :32])
        self.assertAlmostEqual(result, 0.32667673, decimal=6)

        result = loss(targets, logits, weights)
        self.assertAlmostEqual(result, 0.24974126, decimal=6)

        result = loss(targets, logits, weights * 2.0)
        self.assertAlmostEqual(result, 0.24974126 * 2, decimal=6)

    def test_multi(self):
        logits = tf.nn.sigmoid(MULTI_LOGITS)
        targets = tf.one_hot(tf.squeeze(MULTI_TARGETS, -1), 4, dtype="float32")

        loss = MeanSquaredRegressionError()
        result = loss(targets, logits)

        self.assertAlmostEqual(result, 0.42551875, decimal=6)

    def test_batch(self):
        probs = np.random.rand(2, 224, 224, 1).astype("float32")
        targets = np.random.rand(2, 224, 224, 1)

        loss = MeanSquaredRegressionError()
        result0 = loss(targets, probs)
        result1 = (
            sum([loss(targets[i : i + 1], probs[i : i + 1]) for i in range(2)])
            / 2
        )

        self.assertAlmostEqual(result0, result1, decimal=6)

    def test_model(self):
        model = models.Sequential([layers.Dense(5)])
        model.compile(
            loss="SegMe>Loss>MeanSquaredRegressionError",
        )
        model.fit(np.zeros((2, 64, 64, 1)), np.zeros((2, 64, 64, 5), "float32"))
        models.Sequential.from_config(model.get_config())
