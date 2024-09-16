import numpy as np
from keras.src import testing

from segme.common.head import ClassificationActivation
from segme.common.head import ClassificationHead
from segme.common.head import ClassificationUncertainty
from segme.common.head import HeadProjection


class TestHeadProjection(testing.TestCase):
    def test_layer(self):
        self.run_layer_test(
            HeadProjection,
            init_kwargs={"classes": 2, "kernel_size": 3},
            input_shape=(2, 16, 16, 3),
            input_dtype="float32",
            expected_output_shape=(2, 16, 16, 2),
            expected_output_dtype="float32",
        )
        self.run_layer_test(
            HeadProjection,
            init_kwargs={"classes": 4, "kernel_size": 1},
            input_shape=(2, 16, 16, 3),
            input_dtype="float32",
            expected_output_shape=(2, 16, 16, 4),
            expected_output_dtype="float32",
        )


class TestClassificationActivation(testing.TestCase):
    def test_layer(self):
        self.run_layer_test(
            ClassificationActivation,
            init_kwargs={},
            input_shape=(2, 16, 16, 3),
            input_dtype="float32",
            expected_output_shape=(2, 16, 16, 3),
            expected_output_dtype="float32",
        )
        self.run_layer_test(
            ClassificationActivation,
            init_kwargs={},
            input_shape=(2, 16, 16, 1),
            input_dtype="float32",
            expected_output_shape=(2, 16, 16, 1),
            expected_output_dtype="float32",
        )


class TestClassificationHead(testing.TestCase):
    def test_layer(self):
        self.run_layer_test(
            ClassificationHead,
            init_kwargs={
                "classes": 2,
                "kernel_size": 1,
                "kernel_initializer": "variance_scaling",
            },
            input_shape=(2, 16, 16, 3),
            input_dtype="float32",
            expected_output_shape=(2, 16, 16, 2),
            expected_output_dtype="float32",
            run_mixed_precision_check=False,
        )
        self.run_layer_test(
            ClassificationHead,
            init_kwargs={
                "classes": 1,
                "kernel_size": 3,
                "kernel_initializer": "glorot_uniform",
            },
            input_shape=(2, 16, 16, 3),
            input_dtype="float16",
            expected_output_shape=(2, 16, 16, 1),
            expected_output_dtype="float32",
            run_mixed_precision_check=False,
        )
        self.run_layer_test(
            ClassificationHead,
            init_kwargs={
                "classes": 4,
                "kernel_size": 1,
                "kernel_initializer": "glorot_uniform",
            },
            input_shape=(2, 16, 16, 3),
            input_dtype="float16",
            expected_output_shape=(2, 16, 16, 4),
            expected_output_dtype="float32",
            run_mixed_precision_check=False,
        )
        self.run_layer_test(
            ClassificationHead,
            init_kwargs={
                "classes": 4,
                "kernel_size": 1,
                "kernel_initializer": "glorot_uniform",
                "dtype": "mixed_float16",
            },
            input_shape=(2, 16, 16, 3),
            input_dtype="float16",
            expected_output_shape=(2, 16, 16, 4),
            expected_output_dtype="float32",
            run_mixed_precision_check=False,
        )


class TestClassificationUncertainty(testing.TestCase):
    def test_layer(self):
        self.run_layer_test(
            ClassificationUncertainty,
            init_kwargs={"ord": 1, "from_logits": False},
            input_shape=(2, 16, 16, 10),
            input_dtype="float32",
            expected_output_shape=(2, 16, 16, 1),
            expected_output_dtype="float32",
        )
        self.run_layer_test(
            ClassificationUncertainty,
            init_kwargs={"ord": 1, "from_logits": True},
            input_shape=(2, 16, 16, 10),
            input_dtype="float32",
            expected_output_shape=(2, 16, 16, 1),
            expected_output_dtype="float32",
        )
        self.run_layer_test(
            ClassificationUncertainty,
            init_kwargs={"ord": 2, "from_logits": True},
            input_shape=(2, 16, 16, 1),
            input_dtype="float32",
            expected_output_shape=(2, 16, 16, 1),
            expected_output_dtype="float32",
        )

    def test_values_binary_linear(self):
        probs = np.array(
            [
                [
                    [[0.0000000000000000], [0.5953177670614429]],
                    [[0.7229075921797835], [0.6482516524195218]],
                ],
                [
                    [[0.5000000000000000], [0.2893530010764078]],
                    [[0.1822078508133197], [0.4451503035491462]],
                ],
                [
                    [[1.0000000000000000], [0.1093519045907065]],
                    [[0.0554124105966084], [0.7669434550898543]],
                ],
            ]
        )
        expected = np.array(
            [
                [
                    [[0.0000000000000000], [0.8093644380569458]],
                    [[0.5541847944259644], [0.7034966945648193]],
                ],
                [
                    [[1.0000000000000000], [0.5787060260772705]],
                    [[0.36441564559936523], [0.8903006315231323]],
                ],
                [
                    [[0.0000000000000000], [0.2187037467956543]],
                    [[0.1108248233795166], [0.4661130905151367]],
                ],
            ]
        )
        result = ClassificationUncertainty(ord=1, from_logits=False)(probs)
        self.assertAllClose(expected, result)

    def test_values_binary_square(self):
        probs = np.array(
            [
                [
                    [[0.0000000000000000], [0.5953177670614429]],
                    [[0.7229075921797835], [0.6482516524195218]],
                ],
                [
                    [[0.5000000000000000], [0.2893530010764078]],
                    [[0.1822078508133197], [0.4451503035491462]],
                ],
                [
                    [[1.0000000000000000], [0.1093519045907065]],
                    [[0.0554124105966084], [0.7669434550898543]],
                ],
            ]
        )
        expected = np.array(
            [
                [
                    [[0.0000000000000000], [0.9636580944061279]],
                    [[0.8012487888336182], [0.912085771560669]],
                ],
                [
                    [[1.0000000000000000], [0.8225113749504089]],
                    [[0.5960326194763184], [0.987966060638427]],
                ],
                [
                    [[0.0000000000000000], [0.3895762562751770]],
                    [[0.2093674987554550], [0.714964747428894]],
                ],
            ]
        )
        result = ClassificationUncertainty(ord=2, from_logits=False)(probs)
        self.assertAllClose(expected, result)

    def test_values_multiclass_linear(self):
        probs = np.array(
            [
                [
                    [
                        [
                            1.0000000000000000,
                            0.0000000000000000,
                            0.0000000000000000,
                            0.00000000000000000,
                        ],
                        [
                            0.5000000000000000,
                            0.5000000000000000,
                            0.0000000000000000,
                            0.00000000000000000,
                        ],
                    ],
                    [
                        [
                            0.2500000000000000,
                            0.2500000000000000,
                            0.2500000000000000,
                            0.25000000000000000,
                        ],
                        [
                            0.2195614833712468,
                            0.3023602988920650,
                            0.2794063445662053,
                            0.19867187317048300,
                        ],
                    ],
                ],
                [
                    [
                        [
                            0.2632240433966382,
                            0.2708674253716150,
                            0.1995943825396344,
                            0.26631414869211234,
                        ],
                        [
                            0.1997069045070887,
                            0.1702921358306510,
                            0.2873772625954268,
                            0.34262369706683365,
                        ],
                    ],
                    [
                        [
                            0.1801027588111610,
                            0.1709785555646506,
                            0.3345918372338003,
                            0.31432684839038827,
                        ],
                        [
                            0.2493982025135671,
                            0.1939081642271561,
                            0.3468062949178709,
                            0.20988733834140602,
                        ],
                    ],
                ],
                [
                    [
                        [
                            0.2926054786220627,
                            0.2017998186785824,
                            0.3274147121495654,
                            0.17817999054978953,
                        ],
                        [
                            0.1986739712802878,
                            0.1951988303706181,
                            0.3949590413134583,
                            0.21116815703563588,
                        ],
                    ],
                    [
                        [
                            0.2095532140303288,
                            0.1983148473139361,
                            0.3203087849458626,
                            0.27182315370987250,
                        ],
                        [
                            0.3448456980674894,
                            0.1655878692323905,
                            0.1829576333828669,
                            0.30660879931725304,
                        ],
                    ],
                ],
            ]
        )
        expected = np.array(
            [
                [
                    [[0.0000000000000000], [1.0000000000000000]],
                    [[1.0000000000000000], [0.977046012878418]],
                ],
                [
                    [[0.9954466819763184], [0.9447535276412964]],
                    [[0.9797350168228149], [0.902591943740844]],
                ],
                [
                    [[0.9651907682418823], [0.8162091374397278]],
                    [[0.9515143632888794], [0.961763083934783]],
                ],
            ]
        )
        result = ClassificationUncertainty(ord=1, from_logits=False)(probs)
        self.assertAllClose(expected, result)

    def test_values_multiclass_square(self):
        probs = np.array(
            [
                [
                    [
                        [
                            1.0000000000000000,
                            0.0000000000000000,
                            0.0000000000000000,
                            0.00000000000000000,
                        ],
                        [
                            0.5000000000000000,
                            0.5000000000000000,
                            0.0000000000000000,
                            0.00000000000000000,
                        ],
                    ],
                    [
                        [
                            0.2500000000000000,
                            0.2500000000000000,
                            0.2500000000000000,
                            0.25000000000000000,
                        ],
                        [
                            0.2195614833712468,
                            0.3023602988920650,
                            0.2794063445662053,
                            0.19867187317048300,
                        ],
                    ],
                ],
                [
                    [
                        [
                            0.2632240433966382,
                            0.2708674253716150,
                            0.1995943825396344,
                            0.26631414869211234,
                        ],
                        [
                            0.1997069045070887,
                            0.1702921358306510,
                            0.2873772625954268,
                            0.34262369706683365,
                        ],
                    ],
                    [
                        [
                            0.1801027588111610,
                            0.1709785555646506,
                            0.3345918372338003,
                            0.31432684839038827,
                        ],
                        [
                            0.2493982025135671,
                            0.1939081642271561,
                            0.3468062949178709,
                            0.20988733834140602,
                        ],
                    ],
                ],
                [
                    [
                        [
                            0.2926054786220627,
                            0.2017998186785824,
                            0.3274147121495654,
                            0.17817999054978953,
                        ],
                        [
                            0.1986739712802878,
                            0.1951988303706181,
                            0.3949590413134583,
                            0.21116815703563588,
                        ],
                    ],
                    [
                        [
                            0.2095532140303288,
                            0.1983148473139361,
                            0.3203087849458626,
                            0.27182315370987250,
                        ],
                        [
                            0.3448456980674894,
                            0.1655878692323905,
                            0.1829576333828669,
                            0.30660879931725304,
                        ],
                    ],
                ],
            ]
        )
        expected = np.array(
            [
                [
                    [[0.00000000000000000], [1.00000000000000000]],
                    [[0.2500000000000000], [0.337925523519516]],
                ],
                [
                    [[0.28854331374168396], [0.39384907484054565]],
                    [[0.4206847846508026], [0.345971465110779]],
                ],
                [
                    [[0.38321337103843690], [0.33361107110977173]],
                    [[0.3482693731784821], [0.4229309260845184]],
                ],
            ]
        )
        result = ClassificationUncertainty(ord=2, from_logits=False)(probs)
        self.assertAllClose(expected, result)
