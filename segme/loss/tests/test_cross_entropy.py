import numpy as np
from keras.src import ops
from keras.src import testing

from segme.loss.cross_entropy import CrossEntropyLoss
from segme.loss.cross_entropy import cross_entropy_loss
from segme.loss.tests.test_common_loss import BINARY_LOGITS
from segme.loss.tests.test_common_loss import BINARY_TARGETS
from segme.loss.tests.test_common_loss import BINARY_WEIGHTS
from segme.loss.tests.test_common_loss import MULTI_LOGITS
from segme.loss.tests.test_common_loss import MULTI_TARGETS


class TestCrossEntropyLoss(testing.TestCase):
    def test_config(self):
        loss = CrossEntropyLoss(reduction="none", name="loss1")
        self.assertEqual(loss.name, "loss1")
        self.assertEqual(loss.reduction, "none")

    def test_zeros(self):
        logits = -10.0 * ops.ones((3, 64, 64, 1), "float32")
        targets = ops.zeros((3, 64, 64, 1), "int32")

        result = cross_entropy_loss(
            y_true=targets,
            y_pred=logits,
            sample_weight=None,
            from_logits=True,
            label_smoothing=0.0,
            force_binary=False,
        )

        self.assertAllClose(result, [0.0] * 3, atol=6e-3)

    def test_ones(self):
        logits = 10 * ops.ones((3, 64, 64, 1), "float32")
        targets = ops.ones((3, 64, 64, 1), "int32")

        result = cross_entropy_loss(
            y_true=targets,
            y_pred=logits,
            sample_weight=None,
            from_logits=True,
            label_smoothing=0.0,
            force_binary=False,
        )

        self.assertAllClose(result, [0.0] * 3, atol=6e-3)

    def test_false(self):
        logits = -10.0 * ops.ones((3, 64, 64, 1), "float32")
        targets = ops.ones((3, 64, 64, 1), "int32")

        result = cross_entropy_loss(
            y_true=targets,
            y_pred=logits,
            sample_weight=None,
            from_logits=True,
            label_smoothing=0.0,
            force_binary=False,
        )

        self.assertAllClose(result, [10.0] * 3, atol=6e-3)

    def test_true(self):
        logits = 10.0 * ops.ones((3, 64, 64, 1), "float32")
        targets = ops.zeros((3, 64, 64, 1), "int32")

        result = cross_entropy_loss(
            y_true=targets,
            y_pred=logits,
            sample_weight=None,
            from_logits=True,
            label_smoothing=0.0,
            force_binary=False,
        )

        self.assertAllClose(result, [10.0] * 3, atol=6e-3)

    def test_value(self):
        logits = ops.tile(BINARY_LOGITS, [1, 16, 16, 1])
        targets = ops.tile(BINARY_TARGETS, [1, 16, 16, 1])

        loss = CrossEntropyLoss(from_logits=True)
        result = loss(targets, logits)

        self.assertAlmostEqual(result, 1.5399182, decimal=6)

    def test_weight(self):
        logits = ops.tile(BINARY_LOGITS, [1, 16, 16, 1])
        targets = ops.tile(BINARY_TARGETS, [1, 16, 16, 1])
        weights = ops.tile(BINARY_WEIGHTS, [1, 16, 16, 1])

        loss = CrossEntropyLoss(from_logits=True)

        result = loss(targets[:, :, :32], logits[:, :, :32])
        self.assertAlmostEqual(result, 1.5399183, decimal=5)

        result = loss(targets, logits, weights)
        self.assertAlmostEqual(result, 1.0762622, decimal=5)

        result = loss(targets, logits, weights * 2.0)
        self.assertAlmostEqual(result, 1.0762622 * 2, decimal=5)

    def test_multi(self):
        logits = ops.tile(MULTI_LOGITS, [1, 16, 16, 1])
        targets = ops.tile(MULTI_TARGETS, [1, 16, 16, 1])

        loss = CrossEntropyLoss(from_logits=True)
        result = loss(targets, logits)

        self.assertAlmostEqual(result, 5.349822, decimal=5)

    def test_batch(self):
        probs = np.random.rand(2, 224, 224, 1).astype("float32")
        targets = (np.random.rand(2, 224, 224, 1) > 0.5).astype("int32")

        loss = CrossEntropyLoss(from_logits=True)
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
    #         loss="SegMe>Loss>CrossEntropyLoss",
    #     )
    #     model.fit(
    #       ops.zeros((2, 64, 64, 1)), ops.zeros((2, 64, 64, 1), "int32"))
    #     models.Sequential.from_config(model.get_config())
