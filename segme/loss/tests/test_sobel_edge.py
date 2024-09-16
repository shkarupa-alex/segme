import numpy as np
from keras.src import ops
from keras.src import testing

from segme.loss.sobel_edge import SobelEdgeLoss
from segme.loss.sobel_edge import sobel_edge_loss
from segme.loss.tests.test_common_loss import BINARY_LOGITS
from segme.loss.tests.test_common_loss import BINARY_TARGETS
from segme.loss.tests.test_common_loss import BINARY_WEIGHTS
from segme.loss.tests.test_common_loss import MULTI_LOGITS
from segme.loss.tests.test_common_loss import MULTI_TARGETS


class TestSobelEdgeLoss(testing.TestCase):
    def test_config(self):
        loss = SobelEdgeLoss(reduction="none", name="loss1")
        self.assertEqual(loss.name, "loss1")
        self.assertEqual(loss.reduction, "none")

    def test_zeros(self):
        logits = -10.0 * ops.ones((3, 16, 16, 1), "float32")
        targets = ops.zeros((3, 16, 16, 1), "int32")

        result = sobel_edge_loss(
            y_true=targets,
            y_pred=logits,
            sample_weight=None,
            from_logits=True,
            force_binary=False,
        )

        self.assertAllClose(result, [0.0] * 3, atol=1e-4)

    def test_ones(self):
        logits = 10.0 * ops.ones((3, 16, 16, 1), "float32")
        targets = ops.ones((3, 16, 16, 1), "int32")

        result = sobel_edge_loss(
            y_true=targets,
            y_pred=logits,
            sample_weight=None,
            from_logits=True,
            force_binary=False,
        )

        self.assertAllClose(result, [0.0] * 3, atol=1e-4)

    def test_false(self):
        logits = -10.0 * ops.ones((3, 6, 6, 1), "float32")
        targets = ops.ones((3, 6, 6, 1), "int32")

        result = sobel_edge_loss(
            y_true=targets,
            y_pred=logits,
            sample_weight=None,
            from_logits=True,
            force_binary=False,
        )

        self.assertAllClose(result, [0.0] * 3, atol=1e-4)

    def test_true(self):
        logits = 10.0 * ops.ones((3, 6, 6, 1), "float32")
        targets = ops.zeros((3, 6, 6, 1), "int32")

        result = sobel_edge_loss(
            y_true=targets,
            y_pred=logits,
            sample_weight=None,
            from_logits=True,
            force_binary=False,
        )

        self.assertAllClose(result, [0.0] * 3, atol=1e-4)

    def test_value(self):
        loss = SobelEdgeLoss(from_logits=True)
        result = loss(BINARY_TARGETS, BINARY_LOGITS)
        self.assertAlmostEqual(
            result, 0.08220904
        )  # 0.071708525 with zero padding

    def test_weight(self):
        loss = SobelEdgeLoss(from_logits=True)

        result = loss(BINARY_TARGETS[:, :, :2], BINARY_LOGITS[:, :, :2])
        self.assertAlmostEqual(result, 0.08495622)

        result = loss(BINARY_TARGETS, BINARY_LOGITS, BINARY_WEIGHTS)
        self.assertAlmostEqual(result, 0.0928311)

        result = loss(BINARY_TARGETS, BINARY_LOGITS, BINARY_WEIGHTS * 2.0)
        self.assertAlmostEqual(result, 0.0928311 * 2.0)

    def test_multi(self):
        loss = SobelEdgeLoss(from_logits=True)
        result = loss(MULTI_TARGETS, MULTI_LOGITS)
        self.assertAlmostEqual(result, 0.14728217)

    def test_batch(self):
        probs = np.random.rand(2, 224, 224, 1).astype("float32")
        targets = (np.random.rand(2, 224, 224, 1) > 0.5).astype("int32")

        loss = SobelEdgeLoss(from_logits=True)
        result0 = loss(targets, probs)
        result1 = (
            sum([loss(targets[i : i + 1], probs[i : i + 1]) for i in range(2)])
            / 2
        )

        self.assertAlmostEqual(result0, result1)

    # TODO: https://github.com/keras-team/keras/issues/20112
    # def test_model(self):
    #     model = models.Sequential([layers.Dense(5, activation="sigmoid")])
    #     model.compile(
    #         loss="SegMe>Loss>SobelEdgeLoss",
    #     )
    #     model.fit(
    #       ops.zeros((2, 16, 16, 1)), ops.zeros((2, 16, 16, 1), "int32"))
    #     models.Sequential.from_config(model.get_config())
