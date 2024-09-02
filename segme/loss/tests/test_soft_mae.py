import numpy as np
from keras.src import ops
from keras.src import testing

from segme.loss.soft_mae import SoftMeanAbsoluteError
from segme.loss.soft_mae import soft_mean_absolute_error
from segme.loss.tests.test_common_loss import BINARY_LOGITS
from segme.loss.tests.test_common_loss import BINARY_TARGETS
from segme.loss.tests.test_common_loss import BINARY_WEIGHTS
from segme.loss.tests.test_common_loss import MULTI_LOGITS
from segme.loss.tests.test_common_loss import MULTI_TARGETS


class TestSoftMeanAbsoluteError(testing.TestCase):
    def test_config(self):
        loss = SoftMeanAbsoluteError(reduction="none", name="loss1")
        self.assertEqual(loss.name, "loss1")
        self.assertEqual(loss.reduction, "none")

    def test_zeros(self):
        logits = ops.zeros((3, 64, 64, 1), "float32")
        targets = ops.zeros((3, 64, 64, 1), "int32")

        result = soft_mean_absolute_error(
            y_true=targets, y_pred=logits, beta=1.0, sample_weight=None
        )

        self.assertAllClose(result, [0.0] * 3, atol=6e-3)

    def test_ones(self):
        logits = ops.ones((3, 64, 64, 1), "float32")
        targets = ops.ones((3, 64, 64, 1), "int32")

        result = soft_mean_absolute_error(
            y_true=targets, y_pred=logits, beta=1.0, sample_weight=None
        )

        self.assertAllClose(result, [0.0] * 3, atol=6e-3)

    def test_false(self):
        logits = ops.zeros((3, 64, 64, 1), "float32")
        targets = ops.ones((3, 64, 64, 1), "int32")

        result = soft_mean_absolute_error(
            y_true=targets, y_pred=logits, beta=1.0, sample_weight=None
        )

        self.assertAllClose(result, [0.5] * 3, atol=6e-3)

    def test_true(self):
        logits = ops.ones((3, 64, 64, 1), "float32")
        targets = ops.zeros((3, 64, 64, 1), "int32")

        result = soft_mean_absolute_error(
            y_true=targets, y_pred=logits, beta=1.0, sample_weight=None
        )

        self.assertAllClose(result, [0.5] * 3, atol=6e-3)

    def test_value(self):
        logits = np.arange(-10, 11.0)[:, None].astype("float32") / 2.0
        targets = np.zeros_like(logits)
        expected = np.array(
            [
                4.0,
                3.5,
                3.0,
                2.5,
                2.0,
                1.5,
                1.0,
                0.5625,
                0.25,
                0.0625,
                0.0,
                0.0625,
                0.25,
                0.5625,
                1.0,
                1.5,
                2.0,
                2.5,
                3.0,
                3.5,
                4.0,
            ]
        ).astype("float32")

        loss = SoftMeanAbsoluteError(beta=2.0, reduction="none")
        result = loss(targets, logits)

        self.assertAllEqual(result, expected)

    def test_weight(self):
        logits = ops.sigmoid(BINARY_LOGITS)
        targets = ops.cast(BINARY_TARGETS, "float32")
        weights = BINARY_WEIGHTS

        loss = SoftMeanAbsoluteError()

        result = loss(targets[:, :, :32], logits[:, :, :32])
        self.assertAlmostEqual(result, 0.16333841, decimal=6)

        result = loss(targets, logits, weights)
        self.assertAlmostEqual(result, 0.12487066, decimal=6)

        result = loss(targets, logits, weights * 2.0)
        self.assertAlmostEqual(result, 0.12487066 * 2, decimal=6)

    def test_multi(self):
        logits = ops.sigmoid(MULTI_LOGITS)
        targets = ops.one_hot(
            ops.squeeze(MULTI_TARGETS, -1), 4, dtype="float32"
        )

        loss = SoftMeanAbsoluteError()
        result = loss(targets, logits)

        self.assertAlmostEqual(result, 0.2127596, decimal=6)

    def test_batch(self):
        probs = np.random.rand(2, 224, 224, 1).astype("float32")
        targets = np.random.rand(2, 224, 224, 1)

        loss = SoftMeanAbsoluteError()
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
    #         loss="SegMe>Loss>SoftMeanAbsoluteError",
    #     )
    #     model.fit(
    #       np.zeros((2, 64, 64, 5)), np.zeros((2, 64, 64, 5), "float32"))
    #     models.Sequential.from_config(model.get_config())
