import numpy as np
from keras.src import ops
from keras.src import testing

from segme.loss.common_loss import compute_gradient
from segme.loss.common_loss import crossentropy
from segme.loss.common_loss import iou
from segme.loss.common_loss import mae
from segme.loss.common_loss import mse
from segme.loss.common_loss import smooth_labels
from segme.loss.common_loss import to_1hot
from segme.loss.common_loss import to_logits
from segme.loss.common_loss import to_probs
from segme.loss.common_loss import validate_input
from segme.loss.common_loss import weighted_loss


class TestUtils(testing.TestCase):
    def test_validate_input(self):
        targets = (np.random.uniform(size=(2, 4, 4, 1)) > 0.5).astype("int32")
        probs = np.random.uniform(size=(2, 4, 4, 1))
        weights = np.random.uniform(size=(2, 4, 4, 1))

        y_true, y_pred, sample_weight = validate_input(
            targets, probs, weights, dtype=None, rank=None, channel=None
        )
        self.assertAllClose(targets, y_true)
        self.assertAllClose(probs, y_pred)
        self.assertAllClose(weights, sample_weight)

    def test_to_logits(self):
        expected1 = np.array(
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
        expected4 = np.array(
            [
                [
                    [
                        [
                            0.4250706654827763,
                            7.219920928747051,
                            1.14131948950217,
                            2.5576064452206024,
                        ],
                        [
                            1.342442193620409,
                            0.20020616879804165,
                            6.977300484664198,
                            6.280817910206608,
                        ],
                    ],
                    [
                        [
                            0.3206719246447576,
                            0.0176225602425912,
                            1.902292891065069,
                            3.369106587128292,
                        ],
                        [
                            2.6576544216404563,
                            1.863726154333165,
                            4.581314280496405,
                            7.433728759092233,
                        ],
                    ],
                    [
                        [
                            8.13888654097292,
                            1.311411218599392,
                            0.8372454481780323,
                            2.859455217953778,
                        ],
                        [
                            2.0984725413538854,
                            4.619268334888168,
                            8.708732477440673,
                            1.9102341271004541,
                        ],
                    ],
                    [
                        [
                            3.4914178176388266,
                            4.551627675234152,
                            7.709902261544302,
                            3.3982255596983277,
                        ],
                        [
                            0.9182162683255968,
                            7.0387004793287886,
                            2.1883984916630697,
                            1.3921544038795197,
                        ],
                    ],
                ]
            ],
            "float32",
        )

        with self.assertRaisesRegex(ValueError, "Unable to restore logits"):
            to_logits(ops.zeros((1, 2, 2, 1)), from_logits=False)

        with self.assertRaisesRegex(ValueError, "Unable to restore logits"):
            to_logits(ops.zeros((1, 2, 2, 1)), from_logits=False)

        logits1, from_logits1 = to_logits(expected1, from_logits=True)
        self.assertAllClose(logits1, expected1, atol=1e-6)
        self.assertTrue(from_logits1)

        probs1 = ops.sigmoid(expected1)
        probs1._keras_logits = np.array(expected1)
        logits1, from_logits1 = to_logits(probs1, from_logits=False)
        self.assertAllClose(logits1, expected1, atol=1e-6)
        self.assertTrue(from_logits1)

        probs4 = ops.sigmoid(expected4)
        probs4._keras_logits = np.array(expected4)
        logits4, from_logits4 = to_logits(probs4, from_logits=False)
        self.assertAllClose(logits4, expected4, atol=1e-6)
        self.assertTrue(from_logits4)

        with self.assertRaisesRegex(ValueError, "does not represent logits"):
            probs1 = ops.zeros((1, 2, 2, 1))
            probs1._keras_logits = ops.zeros((1, 2, 2, 1))
            to_logits(probs1, from_logits=True)

    def test_to_probs(self):
        logits1 = np.array(
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
        logits4 = np.array(
            [
                [
                    [
                        [
                            0.4250706654827763,
                            7.219920928747051,
                            1.14131948950217,
                            2.5576064452206024,
                        ],
                        [
                            1.342442193620409,
                            0.20020616879804165,
                            6.977300484664198,
                            6.280817910206608,
                        ],
                    ],
                    [
                        [
                            0.3206719246447576,
                            0.0176225602425912,
                            1.902292891065069,
                            3.369106587128292,
                        ],
                        [
                            2.6576544216404563,
                            1.863726154333165,
                            4.581314280496405,
                            7.433728759092233,
                        ],
                    ],
                    [
                        [
                            8.13888654097292,
                            1.311411218599392,
                            0.8372454481780323,
                            2.859455217953778,
                        ],
                        [
                            2.0984725413538854,
                            4.619268334888168,
                            8.708732477440673,
                            1.9102341271004541,
                        ],
                    ],
                    [
                        [
                            3.4914178176388266,
                            4.551627675234152,
                            7.709902261544302,
                            3.3982255596983277,
                        ],
                        [
                            0.9182162683255968,
                            7.0387004793287886,
                            2.1883984916630697,
                            1.3921544038795197,
                        ],
                    ],
                ]
            ],
            "float32",
        )

        probs1, from_logits1 = to_probs(
            logits1, from_logits=True, force_binary=False
        )
        expected1 = ops.sigmoid(logits1)
        self.assertAllClose(probs1, expected1, atol=1e-6)
        self.assertFalse(from_logits1)

        probs4, from_logits4 = to_probs(
            logits4, from_logits=True, force_binary=False
        )
        expected4 = ops.softmax(logits4)
        self.assertAllClose(probs4, expected4, atol=1e-6)
        self.assertFalse(from_logits4)

    def test_to_1hot(self):
        targets1 = np.array(
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
        targets4 = np.array(
            [[[[1], [3]], [[3], [3]], [[1], [2]], [[2], [1]]]], "int32"
        )

        targets1h, _ = to_1hot(
            targets1, np.zeros((2, 4, 4, 1), "float32"), False
        )
        expected1h = ops.concatenate([1 - targets1, targets1], axis=-1)
        self.assertAllClose(targets1h, expected1h, atol=1e-6)

        targets4h, _ = to_1hot(
            targets4, np.zeros((2, 4, 4, 4), "float32"), False
        )
        expected4h = np.array(
            [
                [
                    [[0, 1, 0, 0], [0, 0, 0, 1]],
                    [[0, 0, 0, 1], [0, 0, 0, 1]],
                    [[0, 1, 0, 0], [0, 0, 1, 0]],
                    [[0, 0, 1, 0], [0, 1, 0, 0]],
                ]
            ],
            "int32",
        )
        self.assertAllClose(targets4h, expected4h, atol=4e-6)

    def test_weighted_loss(self):
        loss = np.random.uniform(size=(2, 4, 5, 3))
        weight = np.random.uniform(size=(2, 4, 5, 1)) - 0.5
        weight[weight < 0.0] = 0.0
        expected = np.array(
            [
                (loss[0] * weight[0])[
                    (weight[0] > 0.0).repeat(loss.shape[-1], axis=-1)
                ].mean(),
                (loss[1] * weight[1])[
                    (weight[1] > 0.0).repeat(loss.shape[-1], axis=-1)
                ].mean(),
            ]
        )
        loss, weight = np.array(loss), np.array(weight)

        result = weighted_loss(loss, weight)
        self.assertAllClose(expected, result)

    def test_compute_gradient(self):
        inputs = np.array(
            [
                [
                    [[0.0], [0.0], [1.0], [0.0]],
                    [[1.0], [0.0], [1.0], [1.0]],
                    [[0.0], [1.0], [0.0], [1.0]],
                    [[0.0], [1.0], [1.0], [1.0]],
                ],
                [
                    [[0.0], [1.0], [1.0], [0.0]],
                    [[1.0], [0.0], [0.0], [1.0]],
                    [[0.0], [1.0], [1.0], [0.0]],
                    [[1.0], [1.0], [1.0], [1.0]],
                ],
            ],
            "float32",
        )
        expected_1sub = [
            [
                [[1.0], [0.0], [0.0], [1.0]],
                [[-1.0], [1.0], [-1.0], [0.0]],
                [[0.0], [0.0], [1.0], [0.0]],
            ],
            [
                [[1.0], [-1.0], [-1.0], [1.0]],
                [[-1.0], [1.0], [1.0], [-1.0]],
                [[1.0], [0.0], [0.0], [1.0]],
            ],
        ]
        expected_2min = [
            [
                [[0.0], [0.0], [0.0]],
                [[0.0], [0.0], [1.0]],
                [[0.0], [0.0], [0.0]],
                [[0.0], [1.0], [1.0]],
            ],
            [
                [[0.0], [1.0], [0.0]],
                [[0.0], [0.0], [0.0]],
                [[0.0], [1.0], [0.0]],
                [[1.0], [1.0], [1.0]],
            ],
        ]

        grad_1sub = compute_gradient(inputs, 1, "sub")
        self.assertAllClose(grad_1sub, expected_1sub)

        grad_2min = compute_gradient(inputs, 2, "min")
        self.assertAllClose(grad_2min, expected_2min)

    def test_smooth_labels(self):
        targets_1, logits_1 = to_1hot(BINARY_TARGETS, BINARY_LOGITS, True)
        targets_n, logits_n = to_1hot(MULTI_TARGETS, MULTI_LOGITS, True)

        expected = np.where(targets_1 == 0, 0.05, 0.95)
        result = smooth_labels(targets_1, logits_1, 0.1, False)
        self.assertAllClose(expected, result)

        expected = np.where(targets_n == 0, 0.025, 0.92499995)
        result = smooth_labels(targets_n, logits_n, 0.1, False)
        self.assertAllClose(expected, result)

        expected = np.where(targets_n == 0, 0.05, 0.95)
        result = smooth_labels(targets_n, logits_n, 0.1, True)
        self.assertAllClose(expected, result)


class TestMAE(testing.TestCase):
    def test_zeros(self):
        logits = -10.0 * ops.ones((3, 16, 16, 1), "float32")
        targets = ops.zeros((3, 16, 16, 1), "int32")

        result = mae(
            y_true=targets,
            y_pred=logits,
            sample_weight=None,
            from_logits=True,
            regression=False,
        )
        self.assertAllClose(result, [0.0] * 3, atol=6e-3)

    def test_ones(self):
        logits = 10.0 * ops.ones((3, 16, 16, 1), "float32")
        targets = ops.ones((3, 16, 16, 1), "int32")

        result = mae(
            y_true=targets,
            y_pred=logits,
            sample_weight=None,
            from_logits=True,
            regression=False,
        )
        self.assertAllClose(result, [0.0] * 3, atol=6e-3)

    def test_false(self):
        logits = -10.0 * ops.ones((3, 16, 16, 1), "float32")
        targets = ops.ones((3, 16, 16, 1), "int32")

        result = mae(
            y_true=targets,
            y_pred=logits,
            sample_weight=None,
            from_logits=True,
            regression=False,
        )
        self.assertAllClose(result, [1.0] * 3, atol=6e-3)

    def test_true(self):
        logits = 10 * ops.ones((3, 16, 16, 1), "float32")
        targets = ops.zeros((3, 16, 16, 1), "int32")

        result = mae(
            y_true=targets,
            y_pred=logits,
            sample_weight=None,
            from_logits=True,
            regression=False,
        )
        self.assertAllClose(result, [1.0] * 3, atol=6e-3)

    def test_value(self):
        result = mae(
            y_true=BINARY_TARGETS,
            y_pred=BINARY_LOGITS,
            sample_weight=None,
            from_logits=True,
            regression=False,
        )
        self.assertAllClose(result, [0.375533, 0.417319])

        result = mae(
            y_true=ops.cast(BINARY_TARGETS, "float32"),
            y_pred=ops.sigmoid(BINARY_LOGITS),
            sample_weight=None,
            from_logits=False,
            regression=True,
        )
        self.assertAllClose(result, [0.375533, 0.417319])

    def test_weight(self):
        result = mae(
            y_true=BINARY_TARGETS[:, :, :2],
            y_pred=BINARY_LOGITS[:, :, :2],
            sample_weight=None,
            from_logits=True,
            regression=False,
        )
        self.assertAllClose(result, [0.49504662, 0.17893231])

        result = mae(
            y_true=BINARY_TARGETS,
            y_pred=BINARY_LOGITS,
            sample_weight=BINARY_WEIGHTS,
            from_logits=True,
            regression=False,
        )
        self.assertAllClose(result, [0.49504662, 0.17893231])

        result = mae(
            y_true=BINARY_TARGETS,
            y_pred=BINARY_LOGITS,
            sample_weight=BINARY_WEIGHTS * 2,
            from_logits=True,
            regression=False,
        )
        self.assertAllClose(result, [0.99009323, 0.35786462])

    def test_multi(self):
        result = mae(
            y_true=MULTI_TARGETS,
            y_pred=MULTI_LOGITS,
            sample_weight=None,
            from_logits=True,
            regression=False,
        )
        self.assertAllClose(result, [0.313545])

    def test_smooth(self):
        result = mae(
            y_true=BINARY_TARGETS,
            y_pred=BINARY_LOGITS,
            sample_weight=None,
            from_logits=True,
            regression=False,
            label_smoothing=0.1,
            force_binary=False,
        )
        self.assertAllClose(result, [0.357069, 0.390263])

        result = mae(
            y_true=MULTI_TARGETS,
            y_pred=MULTI_LOGITS,
            sample_weight=None,
            from_logits=True,
            regression=False,
            label_smoothing=0.1,
            force_binary=False,
        )
        self.assertAllClose(result, [0.305073])

        result = mae(
            y_true=MULTI_TARGETS,
            y_pred=MULTI_LOGITS,
            sample_weight=None,
            from_logits=True,
            regression=False,
            label_smoothing=0.1,
            force_binary=True,
        )
        self.assertAllClose(result, [0.48360923])


class TestMSE(testing.TestCase):
    def test_zeros(self):
        logits = -10.0 * ops.ones((3, 16, 16, 1), "float32")
        targets = ops.zeros((3, 16, 16, 1), "int32")

        result = mse(
            y_true=targets,
            y_pred=logits,
            sample_weight=None,
            from_logits=True,
            regression=False,
        )
        self.assertAllClose(result, [0.0] * 3, atol=6e-3)

    def test_ones(self):
        logits = 10.0 * ops.ones((3, 16, 16, 1), "float32")
        targets = ops.ones((3, 16, 16, 1), "int32")

        result = mse(
            y_true=targets,
            y_pred=logits,
            sample_weight=None,
            from_logits=True,
            regression=False,
        )
        self.assertAllClose(result, [0.0] * 3, atol=6e-3)

    def test_false(self):
        logits = -10.0 * ops.ones((3, 16, 16, 1), "float32")
        targets = ops.ones((3, 16, 16, 1), "int32")

        result = mse(
            y_true=targets,
            y_pred=logits,
            sample_weight=None,
            from_logits=True,
            regression=False,
        )
        self.assertAllClose(result, [1.0] * 3, atol=6e-3)

    def test_true(self):
        logits = 10 * ops.ones((3, 16, 16, 1), "float32")
        targets = ops.zeros((3, 16, 16, 1), "int32")

        result = mse(
            y_true=targets,
            y_pred=logits,
            sample_weight=None,
            from_logits=True,
            regression=False,
        )
        self.assertAllClose(result, [1.0] * 3, atol=6e-3)

    def test_value(self):
        result = mse(
            y_true=BINARY_TARGETS,
            y_pred=BINARY_LOGITS,
            sample_weight=None,
            from_logits=True,
            regression=False,
        )
        self.assertAllClose(result, [0.30168968, 0.35166395])

        result = mse(
            y_true=ops.cast(BINARY_TARGETS, "float32"),
            y_pred=ops.sigmoid(BINARY_LOGITS),
            sample_weight=None,
            from_logits=False,
            regression=True,
        )
        self.assertAllClose(result, [0.30168968, 0.35166395])

    def test_weight(self):
        result = mse(
            y_true=BINARY_TARGETS[:, :, :2],
            y_pred=BINARY_LOGITS[:, :, :2],
            sample_weight=None,
            from_logits=True,
            regression=False,
        )
        self.assertAllClose(result, [0.3698082, 0.12967442])

        result = mse(
            y_true=BINARY_TARGETS,
            y_pred=BINARY_LOGITS,
            sample_weight=BINARY_WEIGHTS,
            from_logits=True,
            regression=False,
        )
        self.assertAllClose(result, [0.3698082, 0.12967442])

        result = mse(
            y_true=BINARY_TARGETS,
            y_pred=BINARY_LOGITS,
            sample_weight=BINARY_WEIGHTS * 2,
            from_logits=True,
            regression=False,
        )
        self.assertAllClose(result, [0.7396164, 0.25934884])

    def test_multi(self):
        result = mse(
            y_true=MULTI_TARGETS,
            y_pred=MULTI_LOGITS,
            sample_weight=None,
            from_logits=True,
            regression=False,
        )
        self.assertAllClose(result, [0.269082])

    def test_smooth(self):
        result = mse(
            y_true=BINARY_TARGETS,
            y_pred=BINARY_LOGITS,
            sample_weight=None,
            from_logits=True,
            regression=False,
            label_smoothing=0.1,
            force_binary=False,
        )
        self.assertAllClose(result, [0.266636, 0.312432])

        result = mse(
            y_true=MULTI_TARGETS,
            y_pred=MULTI_LOGITS,
            sample_weight=None,
            from_logits=True,
            regression=False,
            label_smoothing=0.1,
            force_binary=False,
        )
        self.assertAllClose(result, [0.239602])

        result = mse(
            y_true=MULTI_TARGETS,
            y_pred=MULTI_LOGITS,
            sample_weight=None,
            from_logits=True,
            regression=False,
            label_smoothing=0.1,
            force_binary=True,
        )
        self.assertAllClose(result, [0.376386])


class TestCrossentropy(testing.TestCase):
    def test_zeros(self):
        logits = -10.0 * ops.ones((3, 16, 16, 1), "float32")
        targets = ops.zeros((3, 16, 16, 1), "int32")

        result = crossentropy(
            y_true=targets,
            y_pred=logits,
            sample_weight=None,
            from_logits=True,
            label_smoothing=0.0,
            force_binary=False,
        )
        self.assertAllClose(result, [0.0] * 3, atol=6e-3)

    def test_ones(self):
        logits = 10.0 * ops.ones((3, 16, 16, 1), "float32")
        targets = ops.ones((3, 16, 16, 1), "int32")

        result = crossentropy(
            y_true=targets,
            y_pred=logits,
            sample_weight=None,
            from_logits=True,
            label_smoothing=0.0,
            force_binary=False,
        )
        self.assertAllClose(result, [0.0] * 3, atol=6e-3)

    def test_false(self):
        logits = -10.0 * ops.ones((3, 16, 16, 1), "float32")
        targets = ops.ones((3, 16, 16, 1), "int32")

        result = crossentropy(
            y_true=targets,
            y_pred=logits,
            sample_weight=None,
            from_logits=True,
            label_smoothing=0.0,
            force_binary=False,
        )
        self.assertAllClose(result, [10.0] * 3, atol=6e-3)

    def test_true(self):
        logits = 10.0 * ops.ones((3, 16, 16, 1), "float32")
        targets = ops.zeros((3, 16, 16, 1), "int32")

        result = crossentropy(
            y_true=targets,
            y_pred=logits,
            sample_weight=None,
            from_logits=True,
            label_smoothing=0.0,
            force_binary=False,
        )
        self.assertAllClose(result, [10.0] * 3, atol=6e-3)

    def test_value(self):
        result = crossentropy(
            y_true=BINARY_TARGETS,
            y_pred=BINARY_LOGITS,
            sample_weight=None,
            from_logits=True,
            label_smoothing=0.0,
            force_binary=False,
        )
        self.assertAllClose(result, [1.2658163, 1.8140206])

    def test_value_smooth(self):
        result = crossentropy(
            y_true=BINARY_TARGETS,
            y_pred=BINARY_LOGITS,
            sample_weight=None,
            from_logits=True,
            label_smoothing=0.05,
            force_binary=False,
        )
        self.assertAllClose(result, [1.3035736, 1.8281653])

    def test_weight(self):
        result = crossentropy(
            y_true=BINARY_TARGETS[:, :, :2],
            y_pred=BINARY_LOGITS[:, :, :2],
            sample_weight=None,
            from_logits=True,
            label_smoothing=0.0,
            force_binary=False,
        )
        self.assertAllClose(result, [1.6474432, 0.50508237])

        result = crossentropy(
            y_true=BINARY_TARGETS,
            y_pred=BINARY_LOGITS,
            sample_weight=BINARY_WEIGHTS,
            from_logits=True,
            label_smoothing=0.0,
            force_binary=False,
        )
        self.assertAllClose(result, [1.6474432, 0.50508237])

        result = crossentropy(
            y_true=BINARY_TARGETS,
            y_pred=BINARY_LOGITS,
            sample_weight=BINARY_WEIGHTS * 2,
            from_logits=True,
            label_smoothing=0.0,
            force_binary=False,
        )
        self.assertAllClose(result, [3.2948864, 1.0101647])

    def test_multi(self):
        result = crossentropy(
            y_true=MULTI_TARGETS,
            y_pred=MULTI_LOGITS,
            sample_weight=None,
            from_logits=True,
            label_smoothing=0.0,
            force_binary=False,
        )
        self.assertAllClose(result, [5.34982])

    def test_multi_binary(self):
        result = crossentropy(
            y_true=MULTI_TARGETS,
            y_pred=MULTI_LOGITS,
            sample_weight=None,
            from_logits=True,
            label_smoothing=0.0,
            force_binary=True,
        )
        self.assertAllClose(result, [7.669404])

    def test_multi_smooth(self):
        result = crossentropy(
            y_true=MULTI_TARGETS,
            y_pred=MULTI_LOGITS,
            sample_weight=None,
            from_logits=True,
            label_smoothing=0.05,
            force_binary=False,
        )
        self.assertAllClose(result, [5.34137])

    def test_multi_binary_smooth(self):
        result = crossentropy(
            y_true=MULTI_TARGETS,
            y_pred=MULTI_LOGITS,
            sample_weight=None,
            from_logits=True,
            label_smoothing=0.05,
            force_binary=True,
        )
        self.assertAllClose(result, [7.6590743])

    def test_multi_1hot(self):
        targets = ops.one_hot(
            ops.squeeze(MULTI_TARGETS, axis=-1), MULTI_LOGITS.shape[-1]
        )
        result = crossentropy(
            y_true=targets,
            y_pred=MULTI_LOGITS,
            sample_weight=None,
            from_logits=True,
            label_smoothing=0.0,
            force_binary=False,
        )
        self.assertAllClose(result, [5.34982])

    def test_multi_1hot_binary(self):
        targets = ops.one_hot(
            ops.squeeze(MULTI_TARGETS, axis=-1), MULTI_LOGITS.shape[-1]
        )
        result = crossentropy(
            y_true=targets,
            y_pred=MULTI_LOGITS,
            sample_weight=None,
            from_logits=True,
            label_smoothing=0.0,
            force_binary=True,
        )
        self.assertAllClose(result, [7.669404])

    def test_multi_1hot_smooth(self):
        targets = ops.one_hot(
            ops.squeeze(MULTI_TARGETS, axis=-1), MULTI_LOGITS.shape[-1]
        )
        result = crossentropy(
            y_true=targets,
            y_pred=MULTI_LOGITS,
            sample_weight=None,
            from_logits=True,
            label_smoothing=0.05,
            force_binary=False,
        )
        self.assertAllClose(result, [5.34137])


class TestIOU(testing.TestCase):
    def test_zeros(self):
        logits = -10.0 * ops.ones((3, 16, 16, 1), "float32")
        targets = ops.zeros((3, 16, 16, 1), "int32")

        result = iou(
            y_true=targets, y_pred=logits, sample_weight=None, from_logits=True
        )
        self.assertAllClose(result, [0.0] * 3, atol=6e-3)

    def test_ones(self):
        logits = 10.0 * ops.ones((3, 16, 16, 1), "float32")
        targets = ops.ones((3, 16, 16, 1), "int32")

        result = iou(
            y_true=targets, y_pred=logits, sample_weight=None, from_logits=True
        )
        self.assertAllClose(result, [0.0] * 3, atol=6e-3)

    def test_false(self):
        logits = -10.0 * ops.ones((3, 16, 16, 1), "float32")
        targets = ops.ones((3, 16, 16, 1), "int32")

        result = iou(
            y_true=targets, y_pred=logits, sample_weight=None, from_logits=True
        )
        self.assertAllClose(result, [1.0] * 3, atol=6e-3)

    def test_true(self):
        logits = 10 * ops.ones((3, 16, 16, 1), "float32")
        targets = ops.zeros((3, 16, 16, 1), "int32")

        result = iou(
            y_true=targets, y_pred=logits, sample_weight=None, from_logits=True
        )
        self.assertAllClose(result, [1.0] * 3, atol=6e-3)

    def test_value(self):
        result = iou(
            y_true=BINARY_TARGETS,
            y_pred=BINARY_LOGITS,
            sample_weight=None,
            from_logits=True,
        )
        self.assertAllClose(result, [0.5122354, 0.5654068])

    def test_weight(self):
        result = iou(
            y_true=BINARY_TARGETS[:, :, :2],
            y_pred=BINARY_LOGITS[:, :, :2],
            sample_weight=None,
            from_logits=True,
        )
        self.assertAllClose(result, [0.56775665, 0.263336])

        result = iou(
            y_true=BINARY_TARGETS,
            y_pred=BINARY_LOGITS,
            sample_weight=BINARY_WEIGHTS,
            from_logits=True,
        )
        self.assertAllClose(result, [0.61162996, 0.29159677])

        result = iou(
            y_true=BINARY_TARGETS,
            y_pred=BINARY_LOGITS,
            sample_weight=BINARY_WEIGHTS * 2,
            from_logits=True,
        )
        self.assertAllClose(result, [0.6362138, 0.30826524])

    def test_multi(self):
        result = iou(
            y_true=MULTI_TARGETS,
            y_pred=MULTI_LOGITS,
            sample_weight=None,
            from_logits=True,
        )
        self.assertAllClose(result, [0.595264])

    def test_smooth(self):
        result = iou(
            y_true=BINARY_TARGETS,
            y_pred=BINARY_LOGITS,
            sample_weight=None,
            from_logits=True,
            label_smoothing=0.1,
            force_binary=False,
        )
        self.assertAllClose(result, [0.512235, 0.565407])

        result = iou(
            y_true=MULTI_TARGETS,
            y_pred=MULTI_LOGITS,
            sample_weight=None,
            from_logits=True,
            label_smoothing=0.1,
            force_binary=False,
        )
        self.assertAllClose(result, [0.595264])

        result = iou(
            y_true=MULTI_TARGETS,
            y_pred=MULTI_LOGITS,
            sample_weight=None,
            from_logits=True,
            label_smoothing=0.1,
            force_binary=True,
        )
        self.assertAllClose(result, [0.680375])


class TestDice(testing.TestCase):
    def test_zeros(self):
        logits = -10.0 * ops.ones((3, 16, 16, 1), "float32")
        targets = ops.zeros((3, 16, 16, 1), "int32")

        result = iou(
            y_true=targets,
            y_pred=logits,
            sample_weight=None,
            from_logits=True,
            dice=True,
        )
        self.assertAllClose(result, [0.0] * 3, atol=6e-3)

    def test_ones(self):
        logits = 10.0 * ops.ones((3, 16, 16, 1), "float32")
        targets = ops.ones((3, 16, 16, 1), "int32")

        result = iou(
            y_true=targets,
            y_pred=logits,
            sample_weight=None,
            from_logits=True,
            dice=True,
        )
        self.assertAllClose(result, [0.0] * 3, atol=6e-3)

    def test_false(self):
        logits = ops.ones((3, 16, 16, 1), "float32") * (-10.0)
        targets = ops.ones((3, 16, 16, 1), "int32")

        result = iou(
            y_true=targets,
            y_pred=logits,
            sample_weight=None,
            from_logits=True,
            dice=True,
        )
        self.assertAllClose(result, [1.0] * 3, atol=6e-3)

    def test_true(self):
        logits = 10.0 * ops.ones((3, 16, 16, 1), "float32")
        targets = ops.zeros((3, 16, 16, 1), "int32")

        result = iou(
            y_true=targets,
            y_pred=logits,
            sample_weight=None,
            from_logits=True,
            dice=True,
        )
        self.assertAllClose(result, [1.0] * 3, atol=6e-3)

    def test_value(self):
        result = iou(
            y_true=BINARY_TARGETS,
            y_pred=BINARY_LOGITS,
            sample_weight=None,
            from_logits=True,
            dice=True,
        )
        self.assertAllClose(result, [0.37031713, 0.43179172])

    def test_weight(self):
        result = iou(
            y_true=BINARY_TARGETS[:, :, :2],
            y_pred=BINARY_LOGITS[:, :, :2],
            sample_weight=None,
            from_logits=True,
            dice=True,
        )
        self.assertAllClose(result, [0.44075716, 0.17269272])

        result = iou(
            y_true=BINARY_TARGETS,
            y_pred=BINARY_LOGITS,
            sample_weight=BINARY_WEIGHTS,
            from_logits=True,
            dice=True,
        )
        self.assertAllClose(result, [0.46677598, 0.18477038])

        result = iou(
            y_true=BINARY_TARGETS,
            y_pred=BINARY_LOGITS,
            sample_weight=BINARY_WEIGHTS * 2,
            from_logits=True,
            dice=True,
        )
        self.assertAllClose(result, [0.48097467, 0.19151434])

    def test_multi(self):
        result = iou(
            y_true=MULTI_TARGETS,
            y_pred=MULTI_LOGITS,
            sample_weight=None,
            from_logits=True,
            dice=True,
        )
        self.assertAllClose(result, [0.543164])

    def test_smooth(self):
        result = iou(
            y_true=BINARY_TARGETS,
            y_pred=BINARY_LOGITS,
            sample_weight=None,
            from_logits=True,
            dice=True,
            label_smoothing=0.1,
            force_binary=False,
        )
        self.assertAllClose(result, [0.370317, 0.431792])

        result = iou(
            y_true=MULTI_TARGETS,
            y_pred=MULTI_LOGITS,
            sample_weight=None,
            from_logits=True,
            dice=True,
            label_smoothing=0.1,
            force_binary=False,
        )
        self.assertAllClose(result, [0.543164])

        result = iou(
            y_true=MULTI_TARGETS,
            y_pred=MULTI_LOGITS,
            sample_weight=None,
            from_logits=True,
            dice=True,
            label_smoothing=0.1,
            force_binary=True,
        )
        self.assertAllClose(result, [0.606824])


BINARY_LOGITS = np.array(
    [
        [
            [
                [0.4250706654827763],
                [7.219920928747051],
                [7.14131948950217],
                [-2.5576064452206024],
            ],
            [
                [1.342442193620409],
                [0.20020616879804165],
                [-3.977300484664198],
                [6.280817910206608],
            ],
            [
                [0.3206719246447576],
                [-3.0176225602425912],
                [2.902292891065069],
                [3.369106587128292],
            ],
            [
                [-2.6576544216404563],
                [6.863726154333165],
                [4.581314280496405],
                [7.433728759092233],
            ],
        ],
        [
            [
                [-8.13888654097292],
                [8.311411218599392],
                [0.8372454481780323],
                [2.859455217953778],
            ],
            [
                [2.0984725413538854],
                [-4.619268334888168],
                [8.708732477440673],
                [1.9102341271004541],
            ],
            [
                [3.4914178176388266],
                [4.551627675234152],
                [-7.709902261544302],
                [3.3982255596983277],
            ],
            [
                [0.9182162683255968],
                [3.0387004793287886],
                [2.1883984916630697],
                [-1.3921544038795197],
            ],
        ],
    ],
    "float32",
)
BINARY_TARGETS = np.array(
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
BINARY_WEIGHTS = ops.concatenate(
    [ops.ones((2, 4, 2, 1)), ops.zeros((2, 4, 2, 1))], axis=2
)

MULTI_LOGITS = np.array(
    [
        [
            [
                [
                    0.4250706654827763,
                    -7.219920928747051,
                    -1.14131948950217,
                    2.5576064452206024,
                ],
                [
                    -1.342442193620409,
                    0.20020616879804165,
                    -6.977300484664198,
                    6.280817910206608,
                ],
            ],
            [
                [
                    0.3206719246447576,
                    0.0176225602425912,
                    -1.902292891065069,
                    -3.369106587128292,
                ],
                [
                    -2.6576544216404563,
                    1.863726154333165,
                    4.581314280496405,
                    -7.433728759092233,
                ],
            ],
            [
                [
                    8.13888654097292,
                    1.311411218599392,
                    0.8372454481780323,
                    -2.859455217953778,
                ],
                [
                    -2.0984725413538854,
                    -4.619268334888168,
                    8.708732477440673,
                    1.9102341271004541,
                ],
            ],
            [
                [
                    3.4914178176388266,
                    -4.551627675234152,
                    7.709902261544302,
                    3.3982255596983277,
                ],
                [
                    -0.9182162683255968,
                    -7.0387004793287886,
                    2.1883984916630697,
                    1.3921544038795197,
                ],
            ],
        ]
    ],
    "float32",
)
MULTI_TARGETS = np.array(
    [[[[1], [3]], [[3], [3]], [[1], [2]], [[2], [1]]]], "int32"
)
MULTI_WEIGHTS = ops.concatenate(
    [ops.ones((1, 4, 1, 1)), ops.zeros((1, 4, 1, 1))], axis=2
)
