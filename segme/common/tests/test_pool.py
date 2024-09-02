import numpy as np
from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import testing

from segme.common.pool import AdaptiveAveragePooling
from segme.common.pool import MultiHeadAttentionPooling
from segme.common.pool import SimPool


class TestAdaptiveAveragePooling(testing.TestCase):
    def test_layer(self):
        self.run_layer_test(
            AdaptiveAveragePooling,
            init_kwargs={"output_size": 2},
            input_shape=(2, 16, 16, 3),
            input_dtype="float32",
            expected_output_shape=(2, 2, 2, 3),
            expected_output_dtype="float32",
        )
        self.run_layer_test(
            AdaptiveAveragePooling,
            init_kwargs={"output_size": (4, 3)},
            input_shape=(2, 15, 16, 3),
            input_dtype="float32",
            expected_output_shape=(2, 4, 3, 3),
            expected_output_dtype="float32",
        )

    def test_value(self):
        shape = [2, 16, 16, 3]
        data = np.arange(np.prod(shape)).reshape(shape).astype("float32")

        result = AdaptiveAveragePooling(1)(data)
        result = backend.convert_to_numpy(result).astype("int32")
        self.assertListEqual(
            result.ravel().tolist(), [382, 383, 384, 1150, 1151, 1152]
        )

        result = AdaptiveAveragePooling(2)(data)
        result = backend.convert_to_numpy(result).astype("int32")
        self.assertListEqual(
            result.ravel().tolist(),
            [
                178,
                179,
                180,
                202,
                203,
                204,
                562,
                563,
                564,
                586,
                587,
                588,
                946,
                947,
                948,
                970,
                971,
                972,
                1330,
                1331,
                1332,
                1354,
                1355,
                1356,
            ],
        )

        result = AdaptiveAveragePooling(3)(data)
        result = backend.convert_to_numpy(result).astype("int32")
        self.assertListEqual(
            result.ravel().tolist(),
            [
                127,
                128,
                129,
                142,
                143,
                144,
                157,
                158,
                159,
                367,
                368,
                369,
                382,
                383,
                384,
                397,
                398,
                399,
                607,
                608,
                609,
                622,
                623,
                624,
                637,
                638,
                639,
                895,
                896,
                897,
                910,
                911,
                912,
                925,
                926,
                927,
                1135,
                1136,
                1137,
                1150,
                1151,
                1152,
                1165,
                1166,
                1167,
                1375,
                1376,
                1377,
                1390,
                1391,
                1392,
                1405,
                1406,
                1407,
            ],
        )

    def test_placeholder(self):
        shape = [2, 16, 16, 3]
        data = np.arange(np.prod(shape)).reshape(shape).astype("float32")

        inputs = layers.Input([None, None, 3], dtype="float32")
        outputs = AdaptiveAveragePooling(3)(inputs)
        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile("adam", "mse")
        model(data)


class TestMultiheadAttentionPooling(testing.TestCase):
    def test_layer(self):
        self.run_layer_test(
            MultiHeadAttentionPooling,
            init_kwargs={"heads": 8, "queries": 1},
            input_shape=(2, 50, 768),
            input_dtype="float32",
            expected_output_shape=(2, 1, 768),
            expected_output_dtype="float32",
        )
        self.run_layer_test(
            MultiHeadAttentionPooling,
            init_kwargs={"heads": 8, "queries": 2},
            input_shape=(2, 7, 7, 768),
            input_dtype="float32",
            expected_output_shape=(2, 2, 768),
            expected_output_dtype="float32",
        )


class TestSimPool(testing.TestCase):
    def test_layer(self):
        self.run_layer_test(
            SimPool,
            init_kwargs={"num_heads": 1, "qkv_bias": True},
            input_shape=(2, 16, 16, 4),
            input_dtype="float32",
            expected_output_shape=(2, 4),
            expected_output_dtype="float32",
        )
        self.run_layer_test(
            SimPool,
            init_kwargs={"num_heads": 4, "qkv_bias": False},
            input_shape=(2, 16, 16, 4),
            input_dtype="float32",
            expected_output_shape=(2, 4),
            expected_output_dtype="float32",
        )
