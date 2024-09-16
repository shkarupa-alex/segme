import cv2
import numpy as np
from keras.src import ops
from keras.src import testing

from segme.loss.hard_grad import HardGradientMeanAbsoluteError
from segme.loss.hard_grad import hard_gradient_mean_absolute_error


class TestHardGradientMeanAbsoluteError(testing.TestCase):
    def test_config(self):
        loss = HardGradientMeanAbsoluteError(reduction="none", name="loss1")
        self.assertEqual(loss.name, "loss1")
        self.assertEqual(loss.reduction, "none")

    def test_zeros(self):
        probs = ops.zeros((3, 16, 16, 1), "float32")
        targets = ops.zeros((3, 16, 16, 1), "int32")

        result = hard_gradient_mean_absolute_error(
            y_true=targets, y_pred=probs, sample_weight=None
        )

        self.assertAllClose(result, [0.0] * 3, atol=1e-4)

    def test_ones(self):
        probs = ops.ones((3, 16, 16, 1), "float32")
        targets = ops.ones((3, 16, 16, 1), "int32")

        result = hard_gradient_mean_absolute_error(
            y_true=targets, y_pred=probs, sample_weight=None
        )

        self.assertAllClose(result, [0.0] * 3, atol=1e-4)

    def test_false(self):
        probs = ops.zeros((3, 16, 16, 1), "float32")
        targets = ops.ones((3, 16, 16, 1), "int32")

        result = hard_gradient_mean_absolute_error(
            y_true=targets, y_pred=probs, sample_weight=None
        )

        self.assertAllClose(result, [0.0] * 3, atol=1e-4)

    def test_true(self):
        probs = ops.ones((3, 16, 16, 1), "float32")
        targets = ops.zeros((3, 16, 16, 1), "int32")

        result = hard_gradient_mean_absolute_error(
            y_true=targets, y_pred=probs, sample_weight=None
        )

        self.assertAllClose(result, [0.0] * 3, atol=1e-4)

    def test_value(self):
        targets = (
            np.array(
                [
                    [1, 2, 0, 0, 0, 0, 0, 0, 0],
                    [0, 3, 4, 5, 6, 0, 0, 0, 0],
                    [0, 0, 0, 0, 7, 8, 9, 8, 0],
                    [0, 0, 0, 0, 0, 0, 0, 7, 0],
                    [0, 2, 1, 2, 3, 4, 5, 6, 0],
                    [0, 3, 0, 0, 0, 0, 0, 0, 0],
                    [0, 4, 0, 6, 5, 4, 3, 2, 1],
                    [0, 5, 0, 0, 0, 0, 0, 0, 2],
                    [0, 6, 7, 8, 9, 8, 7, 0, 3],
                    [0, 0, 0, 0, 0, 0, 7, 5, 4],
                ]
            ).astype("float32")
            / 9.0
        )
        probs = (targets * 1.9921875) ** 2 / 3.97
        trim = np.where(
            cv2.dilate(targets, np.ones((2, 2), "float32")) > 0, 1.0, 0.0
        )

        loss = HardGradientMeanAbsoluteError()
        result = loss(
            targets[None, ..., None],
            probs[None, ..., None],
            trim[None, ..., None],
        )

        self.assertAlmostEqual(result, 0.20360947)

    def test_weight(self):
        logits = np.array(
            [
                [
                    [
                        [0.4250706654827763],
                        [7.219920928747051],
                        [7.14131948950217],
                        [2.5576064452206024],
                    ],
                    [
                        [1.342442193620409],
                        [0.20020616879804165],
                        [3.977300484664198],
                        [6.280817910206608],
                    ],
                    [
                        [0.3206719246447576],
                        [3.0176225602425912],
                        [2.902292891065069],
                        [3.369106587128292],
                    ],
                    [
                        [2.6576544216404563],
                        [6.863726154333165],
                        [4.581314280496405],
                        [7.433728759092233],
                    ],
                ],
                [
                    [
                        [8.13888654097292],
                        [8.311411218599392],
                        [0.8372454481780323],
                        [2.859455217953778],
                    ],
                    [
                        [2.0984725413538854],
                        [4.619268334888168],
                        [8.708732477440673],
                        [1.9102341271004541],
                    ],
                    [
                        [3.4914178176388266],
                        [4.551627675234152],
                        [7.709902261544302],
                        [3.3982255596983277],
                    ],
                    [
                        [0.9182162683255968],
                        [3.0387004793287886],
                        [2.1883984916630697],
                        [1.3921544038795197],
                    ],
                ],
            ],
            "float32",
        )
        targets = np.array(
            [
                [
                    [[0], [0], [1], [0]],
                    [[1], [0], [1], [1]],
                    [[0], [1], [0], [1]],
                    [[0], [1], [1], [1]],
                ],
                [
                    [[0], [1], [1], [0]],
                    [[1], [0], [0], [1]],
                    [[0], [1], [1], [0]],
                    [[1], [1], [1], [1]],
                ],
            ],
            "int32",
        )
        weights = ops.concatenate(
            [ops.ones((2, 4, 2, 1)), ops.zeros((2, 4, 2, 1))], axis=2
        )

        loss = HardGradientMeanAbsoluteError()

        result = loss(targets[:, :, :2], logits[:, :, :2])
        self.assertAlmostEqual(result, 5.0797915, decimal=6)

        result = loss(targets, logits, weights)
        self.assertAlmostEqual(result, 5.0797915, decimal=6)

        result = loss(targets, logits, weights * 2.0)
        self.assertAlmostEqual(result, 5.0797915 * 2.0, decimal=5)

    def test_multi(self):
        logits = np.array(
            [
                [
                    [
                        [0.42, 7.21, 7.14],
                        [7.21, 7.14, 2.55],
                        [7.14, 2.55, 1.34],
                        [2.55, 1.34, 0.20],
                    ],
                    [
                        [1.34, 0.20, 3.97],
                        [0.20, 3.97, 6.28],
                        [3.97, 6.28, 0.32],
                        [6.28, 0.32, 3.01],
                    ],
                    [
                        [0.32, 3.01, 2.90],
                        [3.01, 2.90, 3.36],
                        [2.90, 3.36, 2.65],
                        [3.36, 2.65, 6.86],
                    ],
                    [
                        [2.65, 6.86, 4.58],
                        [6.86, 4.58, 7.43],
                        [4.58, 7.43, 8.13],
                        [7.43, 8.13, 8.31],
                    ],
                ],
                [
                    [
                        [8.13, 8.31, 0.83],
                        [8.31, 0.83, 2.85],
                        [0.83, 2.85, 2.09],
                        [2.85, 2.09, 4.61],
                    ],
                    [
                        [2.09, 4.61, 8.70],
                        [4.61, 8.70, 1.91],
                        [8.70, 1.91, 3.49],
                        [1.91, 3.49, 4.55],
                    ],
                    [
                        [3.49, 4.55, 7.70],
                        [4.55, 7.70, 3.39],
                        [7.70, 3.39, 0.91],
                        [3.39, 0.91, 3.03],
                    ],
                    [
                        [0.91, 3.03, 2.18],
                        [3.03, 2.18, 1.39],
                        [2.18, 1.39, 0.42],
                        [1.39, 0.42, 7.21],
                    ],
                ],
            ],
            "float32",
        )
        targets = np.array(
            [
                [
                    [[0, 0, 1], [0, 1, 0], [1, 0, 1], [0, 1, 0]],
                    [[1, 0, 1], [0, 1, 1], [1, 1, 0], [1, 0, 1]],
                    [[0, 1, 0], [1, 0, 1], [0, 1, 0], [1, 0, 1]],
                    [[0, 1, 1], [1, 1, 1], [1, 1, 0], [1, 0, 1]],
                ],
                [
                    [[0, 1, 1], [1, 1, 0], [1, 0, 1], [0, 1, 0]],
                    [[1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 1]],
                    [[0, 1, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]],
                    [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
                ],
            ],
            "float32",
        )

        loss = HardGradientMeanAbsoluteError()
        result = loss(targets, logits)

        self.assertAlmostEqual(result, 5.507361, decimal=5)

    def test_batch(self):
        probs = np.random.rand(2, 224, 224, 1).astype("float32")
        targets = (np.random.rand(2, 224, 224, 1) > 0.5).astype("int32")

        loss = HardGradientMeanAbsoluteError()
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
    #         loss="SegMe>Loss>HardGradientMeanAbsoluteError",
    #     )
    #     model.fit(
    #       ops.zeros((2, 16, 16, 1)), ops.zeros((2, 16, 16, 1), "int32"))
    #     models.Sequential.from_config(model.get_config())
