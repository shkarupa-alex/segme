import unittest

import numpy as np
import tensorflow as tf
from absl.testing import parameterized
from keras.src import backend, layers
from keras.src import testing

from segme.policy.norm import NORMALIZATIONS
from segme.policy.norm import BatchNorm
from segme.policy.norm import FilterResponseNorm
from segme.policy.norm import GroupNorm
from segme.policy.norm import LayerNorm
from segme.policy.norm import LayerwiseNorm


class TestNormalizationsRegistry(unittest.TestCase):
    def test_filled(self):
        self.assertIn("bn", NORMALIZATIONS)
        self.assertIn("gn", NORMALIZATIONS)


class TestBatchNormalization(testing.TestCase, parameterized.TestCase):
    def test_layer(self):
        self.run_layer_test(
            BatchNorm,
            init_kwargs={"data_format": "channels_last"},
            input_shape=(2, 8, 16, 3),
            input_dtype="float32",
            expected_output_shape=(2, 8, 16, 3),
            expected_output_dtype="float32",
        )
        self.run_layer_test(
            BatchNorm,
            init_kwargs={"data_format": "channels_first"},
            input_shape=(2, 8, 16, 3),
            input_dtype="float32",
            expected_output_shape=(2, 8, 16, 3),
            expected_output_dtype="float32",
        )

    @parameterized.parameters(
        [
            ("channels_last", "float32"),
            ("channels_last", "float16"),
            ("channels_first", "float32"),
            ("channels_first", "float16"),
        ]
    )
    def test_value(self, dformat, dtype):
        inputs = np.random.uniform(size=(2, 4, 4, 6)).astype("float32")

        if "float16" == dtype:
            inputs = inputs.astype(dtype)
            builtin = layers.BatchNormalization(dtype="mixed_float16")
            custom = BatchNorm(data_format=dformat, dtype="mixed_float16")
        else:
            builtin = layers.BatchNormalization()
            custom = BatchNorm(data_format=dformat)

        expected = builtin(inputs)
        expected = backend.convert_to_numpy(expected)

        if "channels_first" == dformat:
            inputs = inputs.transpose(0, 3, 1, 2)
            expected = expected.transpose(0, 3, 1, 2)

        result = custom(inputs)
        self.assertAllClose(expected, result)


class TestLayerNorm(testing.TestCase, parameterized.TestCase):
    def test_layer(self):
        self.run_layer_test(
            LayerNorm,
            init_kwargs={"data_format": "channels_last"},
            input_shape=(2, 8, 16, 3),
            input_dtype="float32",
            expected_output_shape=(2, 8, 16, 3),
            expected_output_dtype="float32",
        )
        self.run_layer_test(
            LayerNorm,
            init_kwargs={"data_format": "channels_first"},
            input_shape=(2, 8, 16, 3),
            input_dtype="float32",
            expected_output_shape=(2, 8, 16, 3),
            expected_output_dtype="float32",
        )

    @parameterized.parameters(
        [
            ("channels_last", "float32"),
            ("channels_last", "float16"),
            ("channels_first", "float32"),
            ("channels_first", "float16"),
        ]
    )
    def test_value(self, dformat, dtype):
        inputs = np.random.uniform(size=(2, 4, 4, 6)).astype("float32")

        if "float16" == dtype:
            inputs = inputs.astype(dtype)
            builtin = layers.LayerNormalization(dtype="mixed_float16")
            custom = LayerNorm(data_format=dformat, dtype="mixed_float16")
        else:
            builtin = layers.LayerNormalization()
            custom = LayerNorm(data_format=dformat)

        expected = builtin(inputs)
        expected = backend.convert_to_numpy(expected)

        if "channels_first" == dformat:
            inputs = inputs.transpose(0, 3, 1, 2)
            expected = expected.transpose(0, 3, 1, 2)

        result = custom(inputs)
        self.assertAllClose(expected, result)

    def test_batch(self):
        inputs = np.random.normal(size=(32, 16, 16, 64)) * 10.0
        layer = LayerNorm()

        expected = layer(inputs, training=True)

        result = layer(inputs, training=False)
        self.assertAllClose(expected, result)

        result = []
        for i in range(inputs.shape[0]):
            result1 = layer(inputs[i : i + 1], training=False)
            result.append(result1)
        result = np.concatenate(result, axis=0)
        self.assertAllClose(expected, result)


class TestLayerwiseNorm(testing.TestCase, parameterized.TestCase):
    def test_layer(self):
        self.run_layer_test(
            LayerwiseNorm,
            init_kwargs={"data_format": "channels_last"},
            input_shape=(2, 8, 16, 3),
            input_dtype="float32",
            expected_output_shape=(2, 8, 16, 3),
            expected_output_dtype="float32",
        )
        self.run_layer_test(
            LayerwiseNorm,
            init_kwargs={"data_format": "channels_first"},
            input_shape=(2, 8, 16, 3),
            input_dtype="float32",
            expected_output_shape=(2, 8, 16, 3),
            expected_output_dtype="float32",
        )

    @parameterized.parameters(
        [
            ("channels_last", "float32"),
            ("channels_last", "float16"),
            ("channels_first", "float32"),
            ("channels_first", "float16"),
        ]
    )
    def test_value(self, dformat, dtype):
        inputs = np.random.uniform(size=(2, 4, 4, 6)).astype("float32")

        if "float16" == dtype:
            inputs = inputs.astype(dtype)
            builtin = layers.LayerNormalization(axis=[1, 2, 3], dtype="mixed_float16")
            custom = LayerwiseNorm(data_format=dformat, dtype="mixed_float16")
        else:
            builtin = layers.LayerNormalization(axis=[1, 2, 3])
            custom = LayerwiseNorm(data_format=dformat)

        expected = builtin(inputs)
        expected = backend.convert_to_numpy(expected)

        if "channels_first" == dformat:
            inputs = inputs.transpose(0, 3, 1, 2)
            expected = expected.transpose(0, 3, 1, 2)

        result = custom(inputs)
        self.assertAllClose(expected, result)

    def test_batch(self):
        inputs = np.random.normal(size=(32, 16, 16, 64)) * 10.0
        layer = LayerwiseNorm()

        expected = layer(inputs, training=True)

        result = layer(inputs, training=False)
        self.assertAllClose(expected, result)

        result = []
        for i in range(inputs.shape[0]):
            result1 = layer(inputs[i : i + 1], training=False)
            result.append(result1)
        result = np.concatenate(result, axis=0)
        self.assertAllClose(expected, result, atol=1.1e-5)


class TestGroupNorm(testing.TestCase, parameterized.TestCase):
    def test_layer(self):
        self.run_layer_test(
            GroupNorm,
            init_kwargs={"data_format": "channels_last"},
            input_shape=(2, 8, 16, 64),
            input_dtype="float32",
            expected_output_shape=(2, 8, 16, 64),
            expected_output_dtype="float32",
        )
        self.run_layer_test(
            GroupNorm,
            init_kwargs={"data_format": "channels_first"},
            input_shape=(2, 8, 16, 64),
            input_dtype="float32",
            expected_output_shape=(2, 8, 16, 64),
            expected_output_dtype="float32",
        )

    @parameterized.parameters(
        [
            ("channels_last", "float32"),
            ("channels_last", "float16"),
            ("channels_first", "float32"),
            ("channels_first", "float16"),
        ]
    )
    def test_value(self, dformat, dtype):
        inputs = np.random.uniform(size=(2, 4, 4, 6)).astype("float32")

        if "float16" == dtype:
            inputs = inputs.astype(dtype)
            builtin = layers.GroupNormalization(2, dtype="mixed_float16")
            custom = GroupNorm(data_format=dformat, dtype="mixed_float16")
        else:
            builtin = layers.GroupNormalization(2)
            custom = GroupNorm(data_format=dformat)

        expected = builtin(inputs)
        expected = backend.convert_to_numpy(expected)

        if "channels_first" == dformat:
            inputs = inputs.transpose(0, 3, 1, 2)
            expected = expected.transpose(0, 3, 1, 2)

        result = custom(inputs)
        self.assertAllClose(expected, result, atol=2e-3)

    def test_groups(self):
        layer = GroupNorm(groups=-1)
        layer.build([2, 4, 4, 15])
        self.assertEqual(layer.groups, 15)

        layer = GroupNorm(groups=3)
        layer.build([2, 4, 4, 15])
        self.assertEqual(layer.groups, 3)

    def test_groups_auto(self):
        layer = GroupNorm()
        layer.build([2, 4, 4, 1])
        self.assertEqual(layer.groups, 1)

        layer = GroupNorm()
        layer.build([2, 4, 4, 2])
        self.assertEqual(layer.groups, 1)

        layer = GroupNorm()
        layer.build([2, 4, 4, 6])
        self.assertEqual(layer.groups, 2)

        layer = GroupNorm()
        layer.build([2, 4, 4, 8])
        self.assertEqual(layer.groups, 2)

        layer = GroupNorm()
        layer.build([2, 4, 4, 16])
        self.assertEqual(layer.groups, 4)

        layer = GroupNorm()
        layer.build([2, 4, 4, 32])
        self.assertEqual(layer.groups, 4)

        layer = GroupNorm()
        layer.build([2, 4, 4, 64])
        self.assertEqual(layer.groups, 8)

        layer = GroupNorm()
        layer.build([2, 4, 4, 128])
        self.assertEqual(layer.groups, 8)

        layer = GroupNorm()
        layer.build([2, 4, 4, 256])
        self.assertEqual(layer.groups, 16)

        layer = GroupNorm()
        layer.build([2, 4, 4, 512])
        self.assertEqual(layer.groups, 32)

        layer = GroupNorm()
        layer.build([2, 4, 4, 1024])
        self.assertEqual(layer.groups, 32)

        layer = GroupNorm()
        layer.build([2, 4, 4, 2048])
        self.assertEqual(layer.groups, 32)

    def test_batch(self):
        inputs = np.random.normal(size=(32, 16, 16, 64)) * 10.0
        layer = GroupNorm()

        expected = layer(inputs, training=True)

        result = layer(inputs, training=False)
        self.assertAllClose(expected, result)

        result = []
        for i in range(inputs.shape[0]):
            result1 = layer(inputs[i : i + 1], training=False)
            result.append(result1)
        result = np.concatenate(result, axis=0)
        self.assertAllClose(expected, result)


class TestFilterResponseNorm(testing.TestCase, parameterized.TestCase):
    def test_layer(self):
        self.run_layer_test(
            FilterResponseNorm,
            init_kwargs={"data_format": "channels_last"},
            input_shape=(2, 8, 16, 3),
            input_dtype="float32",
            expected_output_shape=(2, 8, 16, 3),
            expected_output_dtype="float32",
        )
        self.run_layer_test(
            FilterResponseNorm,
            init_kwargs={"data_format": "channels_first"},
            input_shape=(2, 8, 16, 3),
            input_dtype="float32",
            expected_output_shape=(2, 8, 16, 3),
            expected_output_dtype="float32",
        )

    @parameterized.parameters(
        [
            ("channels_last", "float32"),
            ("channels_last", "float16"),
            ("channels_first", "float32"),
            ("channels_first", "float16"),
        ]
    )
    def test_value(self, dformat="channels_last", dtype="float32"):
        inputs = (
            np.array(
                [
                    0.18,
                    0.77,
                    0.59,
                    0.12,
                    0.39,
                    0.15,
                    0.02,
                    0.19,
                    0.81,
                    0.29,
                    0.75,
                    0.95,
                    0.67,
                    0.54,
                    0.17,
                    0.1,
                    0.79,
                    0.96,
                    0.57,
                    0.73,
                    0.98,
                    0.43,
                    0.1,
                    0.95,
                    0.55,
                    0.26,
                    0.14,
                    0.56,
                    0.97,
                    0.54,
                    0.76,
                    0.95,
                    0.58,
                    0.31,
                    0.69,
                    0.96,
                    0.71,
                    0.71,
                    0.92,
                    0.19,
                    0.99,
                    0.4,
                    0.34,
                    0.45,
                    0.34,
                    0.92,
                    0.45,
                    0.87,
                    0.15,
                    0.86,
                    0.89,
                    0.14,
                    0.02,
                    0.32,
                    0.2,
                    0.7,
                    0.11,
                    0.54,
                    0.4,
                    0.43,
                    0.4,
                    0.63,
                    0.84,
                    0.59,
                    0.65,
                    0.07,
                    0.59,
                    0.54,
                    0.7,
                    0.29,
                    1.0,
                    0.44,
                    0.64,
                    0.23,
                    0.1,
                    0.66,
                    0.3,
                    0.05,
                    0.01,
                    0.88,
                    0.6,
                    0.8,
                    0.3,
                    0.73,
                    0.73,
                    0.55,
                    0.6,
                    0.49,
                    0.37,
                    0.21,
                    0.25,
                    0.28,
                    0.11,
                    0.51,
                    0.82,
                    0.03,
                    0.66,
                    0.92,
                    0.94,
                    0.65,
                    0.95,
                    0.89,
                    0.93,
                    0.52,
                    0.58,
                    0.73,
                    0.98,
                    0.55,
                    0.37,
                    0.16,
                    0.56,
                    0.96,
                    0.14,
                    0.95,
                    0.31,
                    0.38,
                    0.12,
                    0.47,
                    0.44,
                    1.0,
                    0.32,
                    0.95,
                    0.44,
                    0.34,
                    0.79,
                    0.18,
                    0.26,
                    0.62,
                    0.13,
                    0.51,
                    0.23,
                    0.94,
                    0.48,
                    0.48,
                    0.43,
                    0.81,
                    0.33,
                    0.42,
                    0.97,
                    0.42,
                    0.8,
                    0.3,
                    0.78,
                    0.98,
                    0.18,
                    0.98,
                    0.5,
                    0.31,
                    0.98,
                    0.12,
                    0.32,
                    0.91,
                    0.8,
                    0.56,
                    0.42,
                    0.09,
                    0.26,
                    0.56,
                    0.94,
                    0.63,
                    0.7,
                    0.19,
                    0.15,
                    0.82,
                    0.88,
                    0.09,
                    0.42,
                    0.88,
                    0.55,
                    0.14,
                    0.53,
                    0.23,
                    0.03,
                    0.68,
                    0.14,
                    0.66,
                    0.18,
                    0.83,
                    0.49,
                    0.83,
                    0.44,
                    0.43,
                    0.75,
                    0.03,
                    0.79,
                    0.01,
                    0.47,
                    0.82,
                    0.42,
                    0.91,
                    0.27,
                    0.03,
                ]
            )
            .reshape((2, 4, 4, 6))
            .astype(dtype)
        )
        expected = (
            np.array(
                [
                    0.3654,
                    1.2314,
                    0.9592,
                    0.2429,
                    0.6109,
                    0.2461,
                    0.0406,
                    0.3038,
                    1.3169,
                    0.587,
                    1.1748,
                    1.5584,
                    1.3602,
                    0.8635,
                    0.2764,
                    0.2024,
                    1.2375,
                    1.5749,
                    1.1572,
                    1.1674,
                    1.5933,
                    0.8704,
                    0.1566,
                    1.5584,
                    1.1166,
                    0.4158,
                    0.2276,
                    1.1336,
                    1.5195,
                    0.8859,
                    1.5429,
                    1.5192,
                    0.9429,
                    0.6275,
                    1.0809,
                    1.5749,
                    1.4414,
                    1.1354,
                    1.4957,
                    0.3846,
                    1.5508,
                    0.6562,
                    0.6902,
                    0.7196,
                    0.5528,
                    1.8623,
                    0.7049,
                    1.4272,
                    0.3045,
                    1.3753,
                    1.4469,
                    0.2834,
                    0.0313,
                    0.525,
                    0.406,
                    1.1194,
                    0.1788,
                    1.0931,
                    0.6266,
                    0.7054,
                    0.8121,
                    1.0075,
                    1.3656,
                    1.1943,
                    1.0182,
                    0.1148,
                    1.1978,
                    0.8635,
                    1.138,
                    0.587,
                    1.5665,
                    0.7218,
                    1.2993,
                    0.3678,
                    0.1626,
                    1.336,
                    0.4699,
                    0.082,
                    0.0203,
                    1.4073,
                    0.9755,
                    1.6194,
                    0.4699,
                    1.1975,
                    1.482,
                    0.8795,
                    0.9755,
                    0.9919,
                    0.5796,
                    0.3445,
                    0.5075,
                    0.4478,
                    0.1788,
                    1.0324,
                    1.2845,
                    0.0492,
                    1.3479,
                    1.3842,
                    1.5121,
                    1.0982,
                    1.5247,
                    1.3422,
                    1.8993,
                    0.7824,
                    0.933,
                    1.2334,
                    1.5728,
                    0.8294,
                    0.7556,
                    0.2407,
                    0.9008,
                    1.6219,
                    0.2247,
                    1.4327,
                    0.6331,
                    0.5718,
                    0.193,
                    0.7941,
                    0.7062,
                    1.5081,
                    0.6535,
                    1.4294,
                    0.7078,
                    0.5744,
                    1.2679,
                    0.2715,
                    0.531,
                    0.9329,
                    0.2091,
                    0.8617,
                    0.3691,
                    1.4176,
                    0.9803,
                    0.7222,
                    0.6917,
                    1.3685,
                    0.5296,
                    0.6334,
                    1.981,
                    0.6319,
                    1.2869,
                    0.5069,
                    1.2518,
                    1.4779,
                    0.3676,
                    1.4745,
                    0.8043,
                    0.5238,
                    1.5728,
                    0.181,
                    0.6535,
                    1.3692,
                    1.2869,
                    0.9461,
                    0.6741,
                    0.1357,
                    0.531,
                    0.8426,
                    1.5121,
                    1.0644,
                    1.1235,
                    0.2865,
                    0.3063,
                    1.2338,
                    1.4155,
                    0.1521,
                    0.6741,
                    1.3271,
                    1.1232,
                    0.2106,
                    0.8525,
                    0.3886,
                    0.0481,
                    1.0255,
                    0.2859,
                    0.993,
                    0.2895,
                    1.4023,
                    0.7864,
                    1.2517,
                    0.8986,
                    0.647,
                    1.2064,
                    0.0507,
                    1.2679,
                    0.0151,
                    0.9598,
                    1.2338,
                    0.6756,
                    1.5375,
                    0.4333,
                    0.0452,
                ]
            )
            .reshape((2, 4, 4, 6))
            .astype(dtype)
        )

        if "float16" == dtype:
            custom = FilterResponseNorm(
                data_format=dformat, dtype="mixed_float16")
        else:
            custom = FilterResponseNorm(data_format=dformat)

        if "channels_first" == dformat:
            inputs = inputs.transpose(0, 3, 1, 2)
            expected = expected.transpose(0, 3, 1, 2)

        result = custom(inputs)
        self.assertAllClose(expected, result, atol=2e-3)

    def test_batch(self):
        inputs = np.random.normal(size=(32, 16, 16, 64)) * 10.0
        layer = FilterResponseNorm()

        expected = layer(inputs, training=True)

        result = layer(inputs, training=False)
        self.assertAllClose(expected, result)

        result = []
        for i in range(inputs.shape[0]):
            result1 = layer(inputs[i : i + 1], training=False)
            result.append(result1)
        result = np.concatenate(result, axis=0)
        self.assertAllClose(expected, result)
