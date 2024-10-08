import numpy as np
from keras.src import ops
from keras.src import testing

from segme.loss.heinsen_tree import HeinsenTreeLoss
from segme.loss.heinsen_tree import heinsen_tree_loss


class TestHeinsenTreeLoss(testing.TestCase):
    def test_config(self):
        loss = HeinsenTreeLoss([], reduction="none", name="loss1")
        self.assertEqual(loss.name, "loss1")
        self.assertEqual(loss.reduction, "none")

    def test_zeros_categorical(self):
        logits = -100.0 * ops.ones((3, 18), "float32")
        targets = ops.zeros((3, 1), "int32")

        result = heinsen_tree_loss(
            y_true=targets,
            y_pred=logits,
            sample_weight=None,
            tree_paths=TREE_PATHS,
            force_binary=False,
            label_smoothing=0.0,
            level_weighting=None,
            from_logits=True,
        )

        self.assertAllClose(result, [2.8903718] * 3, atol=1e-4)

    def test_zeros_binary(self):
        logits = -100.0 * ops.ones((3, 18), "float32")
        targets = ops.zeros((3, 1), "int32")

        result = heinsen_tree_loss(
            y_true=targets,
            y_pred=logits,
            sample_weight=None,
            tree_paths=TREE_PATHS,
            force_binary=True,
            label_smoothing=0.0,
            level_weighting=None,
            from_logits=True,
        )

        self.assertAllClose(result, [100.0] * 3, atol=1e-4)

    def test_ones_categorical(self):
        logits = 100.0 * ops.ones((3, 18), "float32")
        targets = ops.ones((3, 1), "int32")

        result = heinsen_tree_loss(
            y_true=targets,
            y_pred=logits,
            sample_weight=None,
            tree_paths=TREE_PATHS,
            force_binary=False,
            label_smoothing=0.0,
            level_weighting=None,
            from_logits=True,
        )

        self.assertAllClose(result, [1.0986123] * 3, atol=1e-4)

    def test_ones_binary(self):
        logits = 100.0 * ops.ones((3, 18), "float32")
        targets = ops.ones((3, 1), "int32")

        result = heinsen_tree_loss(
            y_true=targets,
            y_pred=logits,
            sample_weight=None,
            tree_paths=TREE_PATHS,
            force_binary=True,
            label_smoothing=0.0,
            level_weighting=None,
            from_logits=True,
        )

        self.assertAllClose(result, [200.0] * 3, atol=1e-4)

    def test_false_categorical(self):
        result = heinsen_tree_loss(
            y_true=TREE_TARGETS,
            y_pred=-TRUE_LOGITS,
            sample_weight=None,
            tree_paths=TREE_PATHS,
            force_binary=False,
            label_smoothing=0.0,
            level_weighting=None,
            from_logits=True,
        )

        self.assertAllClose(
            result, [604.0253, 200.69315, 402.07944, 804.0253], atol=1e-4
        )

    def test_false_binary(self):
        result = heinsen_tree_loss(
            y_true=TREE_TARGETS,
            y_pred=-TRUE_LOGITS,
            sample_weight=None,
            tree_paths=TREE_PATHS,
            force_binary=True,
            label_smoothing=0.0,
            level_weighting=None,
            from_logits=True,
        )

        self.assertAllClose(result, [1600.0, 300.0, 800.0, 1800.0], atol=1e-4)

    def test_true_categorical(self):
        result = heinsen_tree_loss(
            y_true=TREE_TARGETS,
            y_pred=TRUE_LOGITS,
            sample_weight=None,
            tree_paths=TREE_PATHS,
            force_binary=False,
            label_smoothing=0.0,
            level_weighting=None,
            from_logits=True,
        )

        self.assertAllClose(result, [0.0] * 4, atol=1e-4)

    def test_true_binary(self):
        result = heinsen_tree_loss(
            y_true=TREE_TARGETS,
            y_pred=TRUE_LOGITS,
            sample_weight=None,
            tree_paths=TREE_PATHS,
            force_binary=True,
            label_smoothing=0.0,
            level_weighting=None,
            from_logits=True,
        )

        self.assertAllClose(result, [0.0] * 4, atol=1e-4)

    def test_value_categorical(self):
        loss = HeinsenTreeLoss(TREE_PATHS, force_binary=False, from_logits=True)
        result = loss(TREE_TARGETS, TREE_LOGITS)

        self.assertAlmostEqual(result, 11.977245, decimal=6)

    def test_value_binary(self):
        loss = HeinsenTreeLoss(TREE_PATHS, force_binary=True, from_logits=True)
        result = loss(TREE_TARGETS, TREE_LOGITS)

        self.assertAlmostEqual(result, 19.158072, decimal=6)

    def test_value_categorical_smooth(self):
        loss = HeinsenTreeLoss(
            TREE_PATHS,
            force_binary=False,
            label_smoothing=1e-5,
            from_logits=True,
        )
        result = loss(TREE_TARGETS, TREE_LOGITS)

        self.assertAlmostEqual(result, 11.980194, decimal=5)

        loss = HeinsenTreeLoss(
            TREE_PATHS,
            force_binary=False,
            label_smoothing=0.1,
            from_logits=True,
        )
        result = loss(TREE_TARGETS, TREE_LOGITS)

        self.assertAlmostEqual(result, 12.815364, decimal=5)

    def test_value_binary_smooth(self):
        loss = HeinsenTreeLoss(
            TREE_PATHS,
            force_binary=True,
            label_smoothing=1e-5,
            from_logits=True,
        )
        result = loss(TREE_TARGETS, TREE_LOGITS)

        self.assertAlmostEqual(result, 19.16037, decimal=5)

        loss = HeinsenTreeLoss(
            TREE_PATHS, force_binary=True, label_smoothing=0.1, from_logits=True
        )
        result = loss(TREE_TARGETS, TREE_LOGITS)

        self.assertAlmostEqual(result, 26.571712, decimal=5)

    def test_value_level_mean(self):
        loss = HeinsenTreeLoss(
            TREE_PATHS, level_weighting="mean", from_logits=True
        )
        result = loss(TREE_TARGETS, TREE_LOGITS)

        self.assertAlmostEqual(result, 4.5775123, decimal=6)

    def test_value_level_linear(self):
        loss = HeinsenTreeLoss(
            TREE_PATHS, level_weighting="linear", from_logits=True
        )
        result = loss(TREE_TARGETS, TREE_LOGITS)

        self.assertAlmostEqual(result, 5.158496, decimal=6)

    def test_value_level_log(self):
        loss = HeinsenTreeLoss(
            TREE_PATHS, level_weighting="log", from_logits=True
        )
        result = loss(TREE_TARGETS, TREE_LOGITS)

        self.assertAlmostEqual(result, 4.924034, decimal=6)

    def test_value_level_pow(self):
        loss = HeinsenTreeLoss(
            TREE_PATHS, level_weighting="pow", from_logits=True
        )
        result = loss(TREE_TARGETS, TREE_LOGITS)

        self.assertAlmostEqual(result, 4.8634114, decimal=6)

    def test_value_level_cumsum(self):
        loss = HeinsenTreeLoss(
            TREE_PATHS, level_weighting="cumsum", from_logits=True
        )
        result = loss(TREE_TARGETS, TREE_LOGITS)

        self.assertAlmostEqual(result, 5.54681, decimal=6)

    def test_value_2d(self):
        loss = HeinsenTreeLoss(TREE_PATHS, from_logits=True)
        targets = ops.reshape(TREE_TARGETS, [2, 2, 1])
        logits = ops.reshape(TREE_LOGITS, [2, 2, 18])
        result = loss(targets, logits)

        self.assertAlmostEqual(result, 11.977245, decimal=6)

    def test_weight(self):
        weights = ops.concatenate([ops.ones((2, 1)), ops.zeros((2, 1))], axis=0)

        loss = HeinsenTreeLoss(TREE_PATHS, from_logits=True)

        result = loss(TREE_TARGETS[:2], TREE_LOGITS[:2])
        self.assertAlmostEqual(result, 7.7723646)

        result = loss(TREE_TARGETS, TREE_LOGITS, weights)
        self.assertAlmostEqual(result, 3.8861823)

        result = loss(TREE_TARGETS, TREE_LOGITS, weights * 2.0)
        self.assertAlmostEqual(result, 3.8861823 * 2.0, decimal=6)

    def test_weight_2d(self):
        targets = ops.reshape(TREE_TARGETS, [2, 2, 1])
        logits = ops.reshape(TREE_LOGITS, [2, 2, 18])
        weights = ops.concatenate([ops.ones((1, 2)), ops.zeros((1, 2))], axis=0)

        loss = HeinsenTreeLoss(TREE_PATHS, from_logits=True)

        result = loss(targets[:1], logits[:1])
        self.assertAlmostEqual(result, 7.7723646)

        result = loss(targets, logits, weights)
        self.assertAlmostEqual(result, 3.8861823)

        result = loss(targets, logits, weights * 2.0)
        self.assertAlmostEqual(result, 3.8861823 * 2.0, decimal=6)

    def test_multi_probs(self):
        probs = 1 / (1 + np.exp(-TREE_LOGITS))
        probs = ops.convert_to_tensor(probs)
        probs._keras_logits = TREE_LOGITS

        loss = HeinsenTreeLoss(TREE_PATHS)
        result = loss(TREE_TARGETS, probs)

        self.assertAlmostEqual(result, 11.977245, decimal=6)

    def test_batch(self):
        targets = ops.reshape(TREE_TARGETS, [2, 2, 1])
        logits = ops.reshape(TREE_LOGITS, [2, 2, 18])

        loss = HeinsenTreeLoss(TREE_PATHS, from_logits=True)
        result0 = loss(targets, logits)
        result1 = (
            sum([loss(targets[i : i + 1], logits[i : i + 1]) for i in range(2)])
            / 2
        )

        self.assertAlmostEqual(result0, result1, decimal=6)

    # TODO: https://github.com/keras-team/keras/issues/20112
    # def test_model(self):
    #     model = models.Sequential([layers.Dense(18, activation="sigmoid")])
    #     model.compile(
    #         loss=HeinsenTreeLoss(TREE_PATHS),
    #     )
    #     model.fit(ops.zeros((2, 18)), ops.zeros((2, 1), "int32"))
    #     models.Sequential.from_config(model.get_config())


#     ╭───────┼──────╮
#     0       1      2
#   ╭─┼─╮       ╭────┴────╮
#   3 4 5       6         7
# ╭─┼─╮       ╭─┴─╮    ╭──┼──╮
# 8 9 10      11 12    13 14 15
#                          ╭─┴─╮
#                          16 17
TREE_PATHS = [
    [0],
    [1],
    [2],
    [0, 3],
    [0, 4],
    [0, 5],
    [2, 6],
    [2, 7],
    [0, 3, 8],
    [0, 3, 9],
    [0, 3, 10],
    [2, 6, 11],
    [2, 6, 12],
    [2, 7, 13],
    [2, 7, 14],
    [2, 7, 15],
    [2, 7, 15, 16],
    [2, 7, 15, 17],
]
TREE_LOGITS = np.array(
    [
        [
            -2.5,
            8.2,
            -2.9,
            6.8,
            -0.8,
            -3.5,
            -7.3,
            4.2,
            5.6,
            -3.3,
            -9.0,
            -1.2,
            0.1,
            6.5,
            -6.6,
            -5.5,
            -5.1,
            -4.7,
        ],
        [
            -7.4,
            -1.7,
            1.8,
            -1.6,
            -0.9,
            -0.5,
            -2.4,
            8.1,
            -8.7,
            1.1,
            -7.0,
            3.4,
            4.2,
            2.9,
            -3.0,
            -7.3,
            -2.7,
            3.5,
        ],
        [
            -3.8,
            1.3,
            5.4,
            3.5,
            3.0,
            8.0,
            -1.8,
            8.4,
            -4.6,
            -0.8,
            -7.2,
            -0.7,
            4.2,
            8.8,
            -5.0,
            4.4,
            1.2,
            2.7,
        ],
        [
            -8.0,
            -1.8,
            3.2,
            -4.0,
            0.1,
            -7.5,
            -4.7,
            -4.2,
            -4.5,
            -4.4,
            -8.5,
            -5.4,
            2.7,
            -3.7,
            -4.8,
            0.2,
            6.1,
            -8.6,
        ],
    ],
    "float32",
)
TREE_TARGETS = np.array([[8], [1], [6], [17]], "int32")
TRUE_LOGITS = 100.0 * np.array(
    [
        [1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 9, 9],
        [-1, 1, -1, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
        [-1, -1, 1, -1, -1, -1, 1, -1, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
        [-1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 1],
    ],
    "float32",
)
