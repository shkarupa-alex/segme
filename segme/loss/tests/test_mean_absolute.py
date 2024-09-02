import numpy as np
from keras.src import ops
from keras.src import testing

from segme.loss.mean_absolute import MeanAbsoluteClassificationError
from segme.loss.mean_absolute import MeanAbsoluteRegressionError
from segme.loss.mean_absolute import mean_absolute_classification_error
from segme.loss.mean_absolute import mean_absolute_regression_error
from segme.loss.tests.test_common_loss import BINARY_LOGITS
from segme.loss.tests.test_common_loss import BINARY_TARGETS
from segme.loss.tests.test_common_loss import BINARY_WEIGHTS
from segme.loss.tests.test_common_loss import MULTI_LOGITS
from segme.loss.tests.test_common_loss import MULTI_TARGETS


class TestMeanAbsoluteClassificationError(testing.TestCase):
    def test_config(self):
        loss = MeanAbsoluteClassificationError(reduction="none", name="loss1")
        self.assertEqual(loss.name, "loss1")
        self.assertEqual(loss.reduction, "none")

    def test_zeros(self):
        logits = -10.0 * ops.ones((3, 64, 64, 1), "float32")
        targets = ops.zeros((3, 64, 64, 1), "int32")

        result = mean_absolute_classification_error(
            y_true=targets,
            y_pred=logits,
            sample_weight=None,
            from_logits=True,
            label_smoothing=0.0,
        )

        self.assertAllClose(result, [0.0] * 3, atol=6e-3)

    def test_ones(self):
        logits = 10 * ops.ones((3, 64, 64, 1), "float32")
        targets = ops.ones((3, 64, 64, 1), "int32")

        result = mean_absolute_classification_error(
            y_true=targets,
            y_pred=logits,
            sample_weight=None,
            from_logits=True,
            label_smoothing=0.0,
        )

        self.assertAllClose(result, [0.0] * 3, atol=6e-3)

    def test_false(self):
        logits = -10.0 * ops.ones((3, 64, 64, 1), "float32")
        targets = ops.ones((3, 64, 64, 1), "int32")

        result = mean_absolute_classification_error(
            y_true=targets,
            y_pred=logits,
            sample_weight=None,
            from_logits=True,
            label_smoothing=0.0,
        )

        self.assertAllClose(result, [1.0] * 3, atol=6e-3)

    def test_true(self):
        logits = 10.0 * ops.ones((3, 64, 64, 1), "float32")
        targets = ops.zeros((3, 64, 64, 1), "int32")

        result = mean_absolute_classification_error(
            y_true=targets,
            y_pred=logits,
            sample_weight=None,
            from_logits=True,
            label_smoothing=0.0,
        )

        self.assertAllClose(result, [1.0] * 3, atol=6e-3)

    def test_value(self):
        logits = BINARY_LOGITS
        targets = BINARY_TARGETS

        loss = MeanAbsoluteClassificationError(from_logits=True)
        result = loss(targets, logits)

        self.assertAlmostEqual(result, 0.39642575)

    def test_weight(self):
        logits = BINARY_LOGITS
        targets = BINARY_TARGETS
        weights = BINARY_WEIGHTS

        loss = MeanAbsoluteClassificationError(from_logits=True)

        result = loss(targets[:, :, :32], logits[:, :, :32])
        self.assertAlmostEqual(result, 0.39642578, decimal=6)

        result = loss(targets, logits, weights)
        self.assertAlmostEqual(result, 0.33698955, decimal=6)

        result = loss(targets, logits, weights * 2.0)
        self.assertAlmostEqual(result, 0.33698955 * 2, decimal=6)

    def test_multi(self):
        logits = MULTI_LOGITS
        targets = MULTI_TARGETS

        loss = MeanAbsoluteClassificationError(from_logits=True)
        result = loss(targets, logits)

        self.assertAlmostEqual(result, 0.31354535, decimal=6)

    def test_batch(self):
        probs = np.random.rand(2, 224, 224, 1).astype("float32")
        targets = (np.random.rand(2, 224, 224, 1) > 0.5).astype("int32")

        loss = MeanAbsoluteClassificationError(from_logits=True)
        result0 = loss(targets, probs)
        result1 = (
            sum([loss(targets[i : i + 1], probs[i : i + 1]) for i in range(2)])
            / 2
        )

        self.assertAlmostEqual(result0, result1, decimal=6)

    # TODO: https://github.com/keras-team/keras/issues/20112
    # def test_model(self):
    #     model = models.Sequential([layers.Dense(5, activation="sigmoid")])
    #     model.compile(
    #         loss="SegMe>Loss>MeanAbsoluteClassificationError",
    #     )
    #     model.fit(np.zeros((2, 64, 64, 1)), np.zeros((2, 64, 64, 1), "int32"))
    #     models.Sequential.from_config(model.get_config())


class TestMeanAbsoluteRegressionError(testing.TestCase):
    def test_config(self):
        loss = MeanAbsoluteRegressionError(reduction="none", name="loss1")
        self.assertEqual(loss.name, "loss1")
        self.assertEqual(loss.reduction, "none")

    def test_zeros(self):
        logits = ops.zeros((3, 64, 64, 1), "float32")
        targets = ops.zeros((3, 64, 64, 1), "int32")

        result = mean_absolute_regression_error(
            y_true=targets, y_pred=logits, sample_weight=None
        )

        self.assertAllClose(result, [0.0] * 3, atol=6e-3)

    def test_ones(self):
        logits = ops.ones((3, 64, 64, 1), "float32")
        targets = ops.ones((3, 64, 64, 1), "int32")

        result = mean_absolute_regression_error(
            y_true=targets, y_pred=logits, sample_weight=None
        )

        self.assertAllClose(result, [0.0] * 3, atol=6e-3)

    def test_false(self):
        logits = ops.zeros((3, 64, 64, 1), "float32")
        targets = ops.ones((3, 64, 64, 1), "int32")

        result = mean_absolute_regression_error(
            y_true=targets, y_pred=logits, sample_weight=None
        )

        self.assertAllClose(result, [1.0] * 3, atol=6e-3)

    def test_true(self):
        logits = ops.ones((3, 64, 64, 1), "float32")
        targets = ops.zeros((3, 64, 64, 1), "int32")

        result = mean_absolute_regression_error(
            y_true=targets, y_pred=logits, sample_weight=None
        )

        self.assertAllClose(result, [1.0] * 3, atol=6e-3)

    def test_value(self):
        logits = ops.sigmoid(BINARY_LOGITS)
        targets = ops.cast(BINARY_TARGETS, "float32")

        loss = MeanAbsoluteRegressionError()
        result = loss(targets, logits)

        self.assertAlmostEqual(result, 0.39642575)

    def test_weight(self):
        logits = ops.sigmoid(BINARY_LOGITS)
        targets = ops.cast(BINARY_TARGETS, "float32")
        weights = BINARY_WEIGHTS

        loss = MeanAbsoluteRegressionError()

        result = loss(targets[:, :, :32], logits[:, :, :32])
        self.assertAlmostEqual(result, 0.39642578, decimal=6)

        result = loss(targets, logits, weights)
        self.assertAlmostEqual(result, 0.33698955, decimal=6)

        result = loss(targets, logits, weights * 2.0)
        self.assertAlmostEqual(result, 0.33698955 * 2, decimal=6)

    def test_multi(self):
        logits = ops.sigmoid(MULTI_LOGITS)
        targets = ops.one_hot(
            ops.squeeze(MULTI_TARGETS, -1), 4, dtype="float32"
        )

        loss = MeanAbsoluteRegressionError()
        result = loss(targets, logits)

        self.assertAlmostEqual(result, 0.5163353, decimal=6)

    def test_batch(self):
        probs = np.random.rand(2, 224, 224, 1).astype("float32")
        targets = np.random.rand(2, 224, 224, 1)

        loss = MeanAbsoluteRegressionError()
        result0 = loss(targets, probs)
        result1 = (
            sum([loss(targets[i : i + 1], probs[i : i + 1]) for i in range(2)])
            / 2
        )

        self.assertAlmostEqual(result0, result1, decimal=6)

    # TODO: https://github.com/keras-team/keras/issues/20112
    # def test_model(self):
    #     model = models.Sequential([layers.Dense(5)])
    #     model.compile(
    #         loss="SegMe>Loss>MeanAbsoluteRegressionError",
    #     )
    #     model.fit(
    #       np.zeros((2, 64, 64, 1)), np.zeros((2, 64, 64, 5), "float32"))
    #     models.Sequential.from_config(model.get_config())
