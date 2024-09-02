import numpy as np
from keras.src import ops
from keras.src import testing

from segme.loss.normalized_focal import NormalizedFocalCrossEntropy
from segme.loss.normalized_focal import normalized_focal_cross_entropy
from segme.loss.tests.test_common_loss import BINARY_LOGITS
from segme.loss.tests.test_common_loss import BINARY_TARGETS
from segme.loss.tests.test_common_loss import BINARY_WEIGHTS
from segme.loss.tests.test_common_loss import MULTI_LOGITS
from segme.loss.tests.test_common_loss import MULTI_TARGETS


class TestNormalizedFocalCrossEntropy(testing.TestCase):
    def test_config(self):
        loss = NormalizedFocalCrossEntropy(reduction="none", name="loss1")
        self.assertEqual(loss.name, "loss1")
        self.assertEqual(loss.reduction, "none")

    def test_zeros(self):
        probs = ops.zeros((3, 16, 16, 1), "float32")
        targets = ops.zeros((3, 16, 16, 1), "int32")

        result = normalized_focal_cross_entropy(
            y_true=targets,
            y_pred=probs,
            sample_weight=None,
            gamma=2,
            from_logits=True,
        )

        self.assertAllClose(result, [0.69314694] * 3, atol=1e-4)

    def test_ones(self):
        probs = ops.ones((3, 16, 16, 1), "float32")
        targets = ops.ones((3, 16, 16, 1), "int32")

        result = normalized_focal_cross_entropy(
            y_true=targets,
            y_pred=probs,
            sample_weight=None,
            gamma=2,
            from_logits=True,
        )

        self.assertAllClose(result, [0.3132613] * 3, atol=1e-4)

    def test_false(self):
        probs = ops.zeros((3, 16, 16, 1), "float32")
        targets = ops.ones((3, 16, 16, 1), "int32")

        result = normalized_focal_cross_entropy(
            y_true=targets,
            y_pred=probs,
            sample_weight=None,
            gamma=2,
            from_logits=True,
        )

        self.assertAllClose(result, [0.69314694] * 3, atol=1e-4)

    def test_true(self):
        probs = ops.ones((3, 16, 16, 1), "float32")
        targets = ops.zeros((3, 16, 16, 1), "int32")

        result = normalized_focal_cross_entropy(
            y_true=targets,
            y_pred=probs,
            sample_weight=None,
            gamma=2,
            from_logits=True,
        )

        self.assertAllClose(result, [1.3132614] * 3, atol=1e-4)

    def test_value(self):
        loss = NormalizedFocalCrossEntropy(from_logits=True)
        result = loss(BINARY_TARGETS, BINARY_LOGITS)

        self.assertAlmostEqual(result, 4.1686983)  # Not sure

    def test_weight(self):
        loss = NormalizedFocalCrossEntropy(from_logits=True)

        result = loss(BINARY_TARGETS[:, :, :2], BINARY_LOGITS[:, :, :2])
        self.assertAlmostEqual(result, 3.4507565, decimal=5)

        result = loss(BINARY_TARGETS, BINARY_LOGITS, BINARY_WEIGHTS)
        self.assertAlmostEqual(result, 2.848103, decimal=5)

        result = loss(BINARY_TARGETS, BINARY_LOGITS, BINARY_WEIGHTS * 2.0)
        self.assertAlmostEqual(result, 2.848103 * 2.0, decimal=5)

    def test_multi(self):
        loss = NormalizedFocalCrossEntropy(from_logits=True)
        result = loss(MULTI_TARGETS, MULTI_LOGITS)

        self.assertAlmostEqual(result, 8.575356, decimal=6)

    def test_batch(self):
        probs = np.random.rand(2, 224, 224, 1).astype("float32")
        targets = (np.random.rand(2, 224, 224, 1) > 0.5).astype("int32")

        loss = NormalizedFocalCrossEntropy(from_logits=True)
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
    #         loss="SegMe>Loss>NormalizedFocalCrossEntropy",
    #     )
    #     model.fit(np.zeros((2, 16, 16, 1)), np.zeros((2, 16, 16, 1), "int32"))
    #     models.Sequential.from_config(model.get_config())
