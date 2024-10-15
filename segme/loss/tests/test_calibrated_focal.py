import numpy as np
from keras.src import ops
from keras.src import testing

from segme.loss.calibrated_focal import CalibratedFocalCrossEntropy
from segme.loss.calibrated_focal import calibrated_focal_cross_entropy
from segme.loss.tests.test_common_loss import BINARY_LOGITS
from segme.loss.tests.test_common_loss import BINARY_TARGETS
from segme.loss.tests.test_common_loss import BINARY_WEIGHTS
from segme.loss.tests.test_common_loss import MULTI_LOGITS
from segme.loss.tests.test_common_loss import MULTI_TARGETS


class TestCalibratedFocalCrossEntropy(testing.TestCase):
    def test_config(self):
        loss = CalibratedFocalCrossEntropy(reduction="none", name="loss1")
        self.assertEqual(loss.name, "loss1")
        self.assertEqual(loss.reduction, "none")

    def test_zeros(self):
        logits = -10.0 * ops.ones((3, 16, 16, 1), "float32")
        targets = ops.zeros((3, 16, 16, 1), "int32")

        result = calibrated_focal_cross_entropy(
            y_true=targets,
            y_pred=logits,
            sample_weight=None,
            prob0=0.2,
            prob1=0.5,
            gamma0=5.0,
            gamma1=3.0,
            from_logits=True,
            label_smoothing=0.0,
            force_binary=False,
        )

        self.assertAllClose(result, [0.0] * 3, atol=1e-4)

    def test_ones(self):
        logits = 10.0 * ops.ones((3, 16, 16, 1), "float32")
        targets = ops.ones((3, 16, 16, 1), "int32")

        result = calibrated_focal_cross_entropy(
            y_true=targets,
            y_pred=logits,
            sample_weight=None,
            prob0=0.2,
            prob1=0.5,
            gamma0=5.0,
            gamma1=3.0,
            from_logits=True,
            label_smoothing=0.0,
            force_binary=False,
        )

        self.assertAllClose(result, [0.0] * 3, atol=1e-4)

    def test_false(self):
        logits = -10.0 * ops.ones((3, 16, 16, 1), "float32")
        targets = ops.ones((3, 16, 16, 1), "int32")

        result = calibrated_focal_cross_entropy(
            y_true=targets,
            y_pred=logits,
            sample_weight=None,
            prob0=0.2,
            prob1=0.5,
            gamma0=5.0,
            gamma1=3.0,
            from_logits=True,
            label_smoothing=0.0,
            force_binary=False,
        )

        self.assertAllClose(result, [9.997775] * 3, atol=1e-4)

    def test_true(self):
        logits = 10.0 * ops.ones((3, 16, 16, 1), "float32")
        targets = ops.zeros((3, 16, 16, 1), "int32")

        result = calibrated_focal_cross_entropy(
            y_true=targets,
            y_pred=logits,
            sample_weight=None,
            prob0=0.2,
            prob1=0.5,
            gamma0=5.0,
            gamma1=3.0,
            from_logits=True,
            label_smoothing=0.0,
            force_binary=False,
        )

        self.assertAllClose(result, [9.997775] * 3, atol=1e-4)

    def test_value(self):
        loss = CalibratedFocalCrossEntropy(from_logits=True)
        result = loss(BINARY_TARGETS, BINARY_LOGITS)

        self.assertAlmostEqual(result, 1.3325094, decimal=6)  # Not sure

    def test_weight(self):
        loss = CalibratedFocalCrossEntropy(from_logits=True)

        result = loss(BINARY_TARGETS[:, :, :2, :], BINARY_LOGITS[:, :, :2])
        self.assertAlmostEqual(result, 0.8728758, decimal=6)

        result = loss(BINARY_TARGETS, BINARY_LOGITS, BINARY_WEIGHTS)
        self.assertAlmostEqual(result, 0.8728758, decimal=6)

        result = loss(BINARY_TARGETS, BINARY_LOGITS, BINARY_WEIGHTS * 2.0)
        self.assertAlmostEqual(result, 0.8728758 * 2.0, decimal=6)

    def test_multi(self):
        loss = CalibratedFocalCrossEntropy(from_logits=True)
        result = loss(MULTI_TARGETS, MULTI_LOGITS)

        self.assertAlmostEqual(result, 5.309415, decimal=6)

    def test_batch(self):
        probs = np.random.rand(2, 224, 224, 2).astype("float32")
        targets = (np.random.rand(2, 224, 224, 1) > 0.5).astype("int32")

        loss = CalibratedFocalCrossEntropy(from_logits=True)
        result0 = loss(targets, probs)
        result1 = (
            sum([loss(targets[i : i + 1], probs[i : i + 1]) for i in range(2)])
            / 2
        )

        self.assertAlmostEqual(result0, result1, decimal=6)

    # TODO: https://github.com/keras-team/keras/issues/20112
    # def test_model(self):
    #     model = models.Sequential([layers.Dense(1, activation="sigmoid")])
    #     model.compile(
    #         loss="SegMe>Loss>CalibratedFocalCrossEntropy",
    #     )
    #     model.fit(
    #       ops.zeros((2, 16, 16, 1)), ops.zeros((2, 16, 16, 1), "int32"))
    #     models.Sequential.from_config(model.get_config())
