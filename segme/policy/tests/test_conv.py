import itertools
import unittest

import numpy as np
import tensorflow as tf
from keras.src import layers
from keras.src import testing

from segme.policy.conv import CONVOLUTIONS
from segme.policy.conv import FixedConv
from segme.policy.conv import FixedDepthwiseConv
from segme.policy.conv import SpectralConv
from segme.policy.conv import SpectralDepthwiseConv
from segme.policy.conv import StandardizedConv
from segme.policy.conv import StandardizedDepthwiseConv


class TestConvsRegistry(unittest.TestCase):
    def test_filled(self):
        self.assertIn("conv", CONVOLUTIONS)
        self.assertIn("stdconv", CONVOLUTIONS)


class TestFixedConv(testing.TestCase):
    def test_layer(self):
        self.run_layer_test(
            FixedConv,
            init_kwargs={
                "filters": 4,
                "kernel_size": 3,
                "strides": 1,
                "dilation_rate": 3,
                "padding": "valid",
            },
            input_shape=(2, 16, 16, 8),
            input_dtype="float32",
            expected_output_shape=(2, 10, 10, 4),
            expected_output_dtype="float32",
        )
        self.run_layer_test(
            FixedConv,
            init_kwargs={
                "filters": 4,
                "kernel_size": 3,
                "strides": 2,
                "dilation_rate": 1,
                "padding": "same",
            },
            input_shape=(2, 16, 16, 8),
            input_dtype="float32",
            expected_output_shape=(2, 8, 8, 4),
            expected_output_dtype="float32",
        )

    def test_valid(self):
        for k, s, d in itertools.product(range(1, 9), range(1, 9), range(1, 9)):
            if s > 1 and d > 1:
                continue

            exconv = layers.Conv2D(
                2, k, strides=s, dilation_rate=d, padding="valid"
            )
            exconv.build([None, None, None, 4])

            layer = FixedConv(2, k, strides=s, dilation_rate=d, padding="valid")
            layer.build([None, None, None, 4])
            layer.set_weights(exconv.get_weights())

            for h, w in itertools.product(
                range(128, 128 + 16 + 1), range(128, 128 + 16 + 1)
            ):
                inputs = np.random.normal(size=(2, h, w, 4)) * 10.0
                exshape = exconv.compute_output_shape(inputs.shape)
                exval = exconv(inputs)
                result = layer(inputs)

                self.assertTupleEqual(exshape, tuple(result.shape.as_list()))
                self.assertLess(np.abs(exval - result).max(), 1e-4)

    def test_same(self):
        for k, s in itertools.product(range(2, 9), range(2, 9)):
            exshapeconv = layers.Conv2D(2, k, strides=s, padding="same")

            exvalconv = layers.Conv2D(2, k, strides=s, padding="valid")
            exvalconv.build([None, None, None, 4])

            layer = FixedConv(2, k, strides=s, padding="same")
            layer.build([None, None, None, 4])
            layer.set_weights(exvalconv.get_weights())

            for h, w in itertools.product(
                range(128, 128 + 16 + 1), range(128, 128 + 16 + 1)
            ):
                inputs = np.random.normal(size=(2, h, w, 4)) * 10.0

                paddings = 1 * (k - 1)
                padbefore = min(paddings // 2, max(0, k - s))
                paddings = padbefore, paddings - padbefore
                painputs = np.pad(inputs, ((0, 0), paddings, paddings, (0, 0)))

                if 3 == k and s in {1, 2}:
                    self.assertTupleEqual(paddings, (1, 1))
                if 5 == k and s in {1, 2}:
                    self.assertTupleEqual(paddings, (2, 2))
                if 7 == k and s in {1, 2}:
                    self.assertTupleEqual(paddings, (3, 3))

                exshape = exshapeconv.compute_output_shape(inputs.shape)
                exval = exvalconv(painputs)
                result = layer(inputs)

                self.assertTupleEqual(exshape, tuple(result.shape.as_list()))
                self.assertLess(np.abs(exval - result).max(), 1e-4)

    def test_same_saw_last(self):
        for k, s, d in itertools.product(range(1, 9), range(1, 9), range(1, 9)):
            if s > 1 and d > 1:
                continue

            layer = FixedConv(1, k, strides=s, dilation_rate=d, padding="same")

            for h, w in itertools.product(
                range(128, 128 + 16 + 1), range(128, 128 + 16 + 1)
            ):
                inputs = np.zeros((1, h, w, 4))
                inputs[:, -max(d, s - k + 1) :, -max(d, s - k + 1) :] = 1.0

                result = layer(inputs)
                self.assertNotEqual(result[0, -1, -1, 0], 0.0)

    def test_same_valid_equal(self):
        inputs = np.random.uniform(size=[2, 16, 16, 3]).astype("float32")

        valconv = FixedConv(8, 4, strides=4, padding="valid")
        expected = valconv(inputs)

        sameconv = FixedConv(8, 4, strides=4, padding="same")
        sameconv.build([None, None, None, 3])
        sameconv.set_weights(valconv.get_weights())
        result = sameconv(inputs)

        self.assertLess(np.abs(expected - result).max(), 1e-6)


class TestFixedDepthwiseConv(testing.TestCase):
    def test_layer(self):
        self.run_layer_test(
            FixedDepthwiseConv,
            init_kwargs={
                "kernel_size": 3,
                "strides": 1,
                "dilation_rate": 3,
                "padding": "same",
            },
            input_shape=(2, 16, 16, 8),
            input_dtype="float32",
            expected_output_shape=(2, 16, 16, 8),
            expected_output_dtype="float32",
        )
        self.run_layer_test(
            FixedDepthwiseConv,
            init_kwargs={
                "kernel_size": 3,
                "strides": 2,
                "dilation_rate": 1,
                "padding": "valid",
            },
            input_shape=(2, 16, 16, 8),
            input_dtype="float32",
            expected_output_shape=(2, 7, 7, 8),
            expected_output_dtype="float32",
        )

    def test_valid(self):
        for k, s, d in itertools.product(range(1, 9), range(1, 9), range(1, 9)):
            if s > 1 and d > 1:
                continue

            exconv = layers.DepthwiseConv2D(
                k, strides=s, dilation_rate=d, padding="valid"
            )
            exconv.build([None, None, None, 4])

            layer = FixedDepthwiseConv(
                k, strides=s, dilation_rate=d, padding="valid"
            )
            layer.build([None, None, None, 4])
            layer.set_weights(exconv.get_weights())

            for h, w in itertools.product(
                range(128, 128 + 16 + 1), range(128, 128 + 16 + 1)
            ):
                inputs = np.random.normal(size=(2, h, w, 4)) * 10.0
                exshape = exconv.compute_output_shape(inputs.shape)
                exval = exconv(inputs)
                result = layer(inputs)

                self.assertTupleEqual(exshape, tuple(result.shape.as_list()))
                self.assertLess(np.abs(exval - result).max(), 1e-4)

    def test_same(self):
        for k, s in itertools.product(range(1, 9), range(1, 9)):
            exshapeconv = layers.DepthwiseConv2D(k, strides=s, padding="same")

            exvalconv = layers.DepthwiseConv2D(k, strides=s, padding="valid")
            exvalconv.build([None, None, None, 4])

            layer = FixedDepthwiseConv(k, strides=s, padding="same")
            layer.build([None, None, None, 4])
            layer.set_weights(exvalconv.get_weights())

            for h, w in itertools.product(
                range(128, 128 + 16 + 1), range(128, 128 + 16 + 1)
            ):
                inputs = np.random.normal(size=(2, h, w, 4)) * 10.0

                paddings = 1 * (k - 1)
                padbefore = min(paddings // 2, max(0, k - s))
                paddings = padbefore, paddings - padbefore
                painputs = np.pad(inputs, ((0, 0), paddings, paddings, (0, 0)))

                if 3 == k and s in {1, 2}:
                    self.assertTupleEqual(paddings, (1, 1))
                if 5 == k and s in {1, 2}:
                    self.assertTupleEqual(paddings, (2, 2))
                if 7 == k and s in {1, 2}:
                    self.assertTupleEqual(paddings, (3, 3))

                exshape = exshapeconv.compute_output_shape(inputs.shape)
                exval = exvalconv(painputs)
                result = layer(inputs)

                self.assertTupleEqual(exshape, tuple(result.shape.as_list()))
                self.assertLess(np.abs(exval - result).max(), 1e-4)

    def test_same_saw_last(self):
        for k, s, d in itertools.product(range(1, 9), range(1, 9), range(1, 9)):
            if s > 1 and d > 1:
                continue

            layer = FixedDepthwiseConv(
                k, strides=s, dilation_rate=d, padding="same"
            )

            for h, w in itertools.product(
                range(128, 128 + 16 + 1), range(128, 128 + 16 + 1)
            ):
                inputs = np.zeros((1, h, w, 1))
                inputs[:, -max(d, s - k + 1) :, -max(d, s - k + 1) :] = 1.0

                result = layer(inputs)
                self.assertNotEqual(result[0, -1, -1, 0], 0.0)


class TestStandardizedConv(testing.TestCase):
    def test_layer(self):
        self.run_layer_test(
            StandardizedConv,
            init_kwargs={
                "filters": 4,
                "kernel_size": 1,
                "strides": 1,
                "padding": "valid",
            },
            input_shape=(2, 16, 16, 8),
            input_dtype="float32",
            expected_output_shape=(2, 16, 16, 4),
            expected_output_dtype="float32",
        )
        self.run_layer_test(
            StandardizedConv,
            init_kwargs={
                "filters": 4,
                "kernel_size": 3,
                "strides": 2,
                "padding": "same",
            },
            input_shape=(2, 16, 16, 8),
            input_dtype="float32",
            expected_output_shape=(2, 8, 8, 4),
            expected_output_dtype="float32",
        )

    def test_value(self):
        inputs = np.array(
            [
                0.298,
                -0.336,
                4.725,
                -0.393,
                -1.951,
                -3.269,
                -2.453,
                -0.043,
                0.25,
                -4.571,
                -2.143,
                2.744,
                1.483,
                3.229,
                1.472,
                -1.802,
                3.146,
                -0.048,
                1.407,
                1.315,
                -0.823,
                0.763,
                -0.103,
                0.295,
                1.507,
                -3.52,
                -1.55,
                -2.573,
                0.929,
                1.649,
                1.545,
                -0.365,
                1.845,
                1.208,
                -0.829,
                -3.652,
            ],
            "float32",
        ).reshape((1, 3, 4, 3))
        kernel = np.array(
            [
                0.307,
                -0.094,
                0.031,
                -0.301,
                -0.164,
                -0.073,
                0.07,
                -0.167,
                0.267,
                -0.128,
                -0.226,
                -0.181,
                -0.248,
                -0.05,
                0.056,
                -0.535,
                0.221,
                -0.04,
                0.521,
                -0.285,
                -0.323,
                0.094,
                0.362,
                -0.022,
                -0.097,
                -0.054,
                -0.084,
            ],
            "float32",
        ).reshape((3, 3, 3, 1))
        bias = np.zeros((1,), "float32")
        expected = np.array([7.993624, -9.57225], "float32").reshape(
            (1, 1, 2, 1)
        )

        layer = StandardizedConv(1, 3)
        layer.build(inputs.shape)
        layer.set_weights([kernel, bias])

        result = layer(inputs)
        self.assertAllClose(expected, result)

    def test_value_fp16(self):
        inputs = np.array(
            [
                0.298,
                -0.336,
                4.725,
                -0.393,
                -1.951,
                -3.269,
                -2.453,
                -0.043,
                0.25,
                -4.571,
                -2.143,
                2.744,
                1.483,
                3.229,
                1.472,
                -1.802,
                3.146,
                -0.048,
                1.407,
                1.315,
                -0.823,
                0.763,
                -0.103,
                0.295,
                1.507,
                -3.52,
                -1.55,
                -2.573,
                0.929,
                1.649,
                1.545,
                -0.365,
                1.845,
                1.208,
                -0.829,
                -3.652,
            ],
            "float32",
        ).reshape((1, 3, 4, 3))
        kernel = np.array(
            [
                0.307,
                -0.094,
                0.031,
                -0.301,
                -0.164,
                -0.073,
                0.07,
                -0.167,
                0.267,
                -0.128,
                -0.226,
                -0.181,
                -0.248,
                -0.05,
                0.056,
                -0.535,
                0.221,
                -0.04,
                0.521,
                -0.285,
                -0.323,
                0.094,
                0.362,
                -0.022,
                -0.097,
                -0.054,
                -0.084,
            ],
            "float32",
        ).reshape((3, 3, 3, 1))
        bias = np.zeros((1,), "float32")
        expected = np.array([7.993624, -9.57225], "float32").reshape(
            (1, 1, 2, 1)
        )

        layer = StandardizedConv(1, 3, dtype="mixed_float16")
        layer.build(inputs.shape)
        layer.set_weights([kernel, bias])

        result = layer(inputs)
        self.assertAllClose(expected, result, atol=7e-3)

    def test_batch(self):
        inputs = np.random.normal(size=(32, 16, 16, 64)) * 10.0
        layer = StandardizedConv(64, 3)

        expected = layer(inputs, training=True)

        result = layer(inputs, training=False)
        self.assertAllClose(expected, result)

        result = []
        for i in range(inputs.shape[0]):
            result1 = layer(inputs[i : i + 1], training=False)
            result.append(result1)
        result = np.concatenate(result, axis=0)
        self.assertAllClose(expected, result, atol=3e-4)


class TestStandardizedDepthwiseConv(testing.TestCase):
    def test_layer(self):
        self.run_layer_test(
            StandardizedDepthwiseConv,
            init_kwargs={"kernel_size": 1, "strides": 1, "padding": "valid"},
            input_shape=(2, 16, 16, 8),
            input_dtype="float32",
            expected_output_shape=(2, 16, 16, 8),
            expected_output_dtype="float32",
        )
        self.run_layer_test(
            StandardizedDepthwiseConv,
            init_kwargs={"kernel_size": 3, "strides": 2, "padding": "same"},
            input_shape=(2, 16, 16, 8),
            input_dtype="float32",
            expected_output_shape=(2, 8, 8, 8),
            expected_output_dtype="float32",
        )

    def test_value(self):
        inputs = np.array(
            [
                0.298,
                -0.336,
                4.725,
                -0.393,
                -1.951,
                -3.269,
                -2.453,
                -0.043,
                0.25,
                -4.571,
                -2.143,
                2.744,
                1.483,
                3.229,
                1.472,
                -1.802,
                3.146,
                -0.048,
                1.407,
                1.315,
                -0.823,
                0.763,
                -0.103,
                0.295,
                1.507,
                -3.52,
                -1.55,
                -2.573,
                0.929,
                1.649,
                1.545,
                -0.365,
                1.845,
                1.208,
                -0.829,
                -3.652,
            ],
            "float32",
        ).reshape((1, 3, 4, 3))
        kernel = np.array(
            [
                0.307,
                -0.094,
                0.031,
                -0.301,
                -0.164,
                -0.073,
                0.07,
                -0.167,
                0.267,
                -0.128,
                -0.226,
                -0.181,
                -0.248,
                -0.05,
                0.056,
                -0.535,
                0.221,
                -0.04,
                0.521,
                -0.285,
                -0.323,
                0.094,
                0.362,
                -0.022,
                -0.097,
                -0.054,
                -0.084,
            ],
            "float32",
        ).reshape((3, 3, 3, 1))
        bias = np.zeros((3,), "float32")
        expected = np.array(
            [
                -0.32465044,
                6.2342463,
                4.557652,
                -5.863132,
                -3.0357158,
                1.6686258,
            ],
            "float32",
        ).reshape((1, 1, 2, 3))

        layer = StandardizedDepthwiseConv(3)
        layer.build(inputs.shape)
        layer.set_weights([kernel, bias])

        result = layer(inputs)
        self.assertAllClose(expected, result)

    def test_value_fp16(self):
        inputs = np.array(
            [
                0.298,
                -0.336,
                4.725,
                -0.393,
                -1.951,
                -3.269,
                -2.453,
                -0.043,
                0.25,
                -4.571,
                -2.143,
                2.744,
                1.483,
                3.229,
                1.472,
                -1.802,
                3.146,
                -0.048,
                1.407,
                1.315,
                -0.823,
                0.763,
                -0.103,
                0.295,
                1.507,
                -3.52,
                -1.55,
                -2.573,
                0.929,
                1.649,
                1.545,
                -0.365,
                1.845,
                1.208,
                -0.829,
                -3.652,
            ],
            "float32",
        ).reshape((1, 3, 4, 3))
        kernel = np.array(
            [
                0.307,
                -0.094,
                0.031,
                -0.301,
                -0.164,
                -0.073,
                0.07,
                -0.167,
                0.267,
                -0.128,
                -0.226,
                -0.181,
                -0.248,
                -0.05,
                0.056,
                -0.535,
                0.221,
                -0.04,
                0.521,
                -0.285,
                -0.323,
                0.094,
                0.362,
                -0.022,
                -0.097,
                -0.054,
                -0.084,
            ],
            "float32",
        ).reshape((3, 3, 3, 1))
        bias = np.zeros((3,), "float32")
        expected = np.array(
            [
                -0.32465044,
                6.2342463,
                4.557652,
                -5.863132,
                -3.0357158,
                1.6686258,
            ],
            "float32",
        ).reshape((1, 1, 2, 3))

        layer = StandardizedDepthwiseConv(3, dtype="mixed_float16")
        layer.build(inputs.shape)
        layer.set_weights([kernel, bias])

        result = layer(inputs)
        self.assertAllClose(expected, result, atol=7e-3)

    def test_batch(self):
        inputs = np.random.normal(size=(32, 16, 16, 64)) * 10.0
        layer = StandardizedDepthwiseConv(3)

        expected = layer(inputs, training=True)

        result = layer(inputs, training=False)
        self.assertAllClose(expected, result)

        result = []
        for i in range(inputs.shape[0]):
            result1 = layer(inputs[i : i + 1], training=False)
            result.append(result1)
        result = np.concatenate(result, axis=0)
        self.assertAllClose(expected, result, atol=1e-4)


class TestSpectralConv(testing.TestCase):
    def test_layer(self):
        self.run_layer_test(
            SpectralConv,
            init_kwargs={
                "filters": 4,
                "kernel_size": 1,
                "strides": 1,
                "padding": "valid",
                "power_iterations": 2,
            },
            input_shape=(2, 16, 16, 8),
            input_dtype="float32",
            expected_output_shape=(2, 16, 16, 4),
            expected_output_dtype="float32",
        )
        self.run_layer_test(
            SpectralConv,
            init_kwargs={
                "filters": 4,
                "kernel_size": 3,
                "strides": 2,
                "padding": "same",
                "power_iterations": 1,
            },
            input_shape=(2, 16, 16, 8),
            input_dtype="float32",
            expected_output_shape=(2, 8, 8, 4),
            expected_output_dtype="float32",
        )

    def test_value(self):
        inputs = np.array(
            [
                0.298,
                -0.336,
                4.725,
                -0.393,
                -1.951,
                -3.269,
                -2.453,
                -0.043,
                0.25,
                -4.571,
                -2.143,
                2.744,
                1.483,
                3.229,
                1.472,
                -1.802,
                3.146,
                -0.048,
                1.407,
                1.315,
                -0.823,
                0.763,
                -0.103,
                0.295,
                1.507,
                -3.52,
                -1.55,
                -2.573,
                0.929,
                1.649,
                1.545,
                -0.365,
                1.845,
                1.208,
                -0.829,
                -3.652,
            ],
            "float32",
        ).reshape((1, 3, 4, 3))
        kernel = np.array(
            [
                0.307,
                -0.094,
                0.031,
                -0.301,
                -0.164,
                -0.073,
                0.07,
                -0.167,
                0.267,
                -0.128,
                -0.226,
                -0.181,
                -0.248,
                -0.05,
                0.056,
                -0.535,
                0.221,
                -0.04,
                0.521,
                -0.285,
                -0.323,
                0.094,
                0.362,
                -0.022,
                -0.097,
                -0.054,
                -0.084,
            ],
            "float32",
        ).reshape((3, 3, 3, 1))
        bias = np.zeros((1,), "float32")
        u = np.array([[-0.008]], "float32")
        expected = np.array([1.3133075, -1.533053], "float32").reshape(
            (1, 1, 2, 1)
        )

        layer = SpectralConv(1, 3)
        layer.build(inputs.shape)
        layer.set_weights([kernel, bias, u])

        result = layer(inputs, training=True)
        self.assertAllClose(expected, result)

        result = layer(inputs)
        self.assertAllClose(expected, result)

    def test_batch(self):
        inputs = np.random.normal(size=(32, 16, 16, 64)) * 10.0
        layer = SpectralConv(64, 3)

        expected = layer(inputs, training=True)

        result = layer(inputs, training=False)
        self.assertAllClose(expected, result)

        result = []
        for i in range(inputs.shape[0]):
            result1 = layer(inputs[i : i + 1], training=False)
            result.append(result1)
        result = np.concatenate(result, axis=0)
        self.assertAllClose(expected, result, atol=6e-5)


class TestSpectralDepthwiseConv(testing.TestCase):
    def test_layer(self):
        self.run_layer_test(
            SpectralDepthwiseConv,
            init_kwargs={
                "kernel_size": 1,
                "strides": 1,
                "padding": "valid",
                "power_iterations": 2,
            },
            input_shape=(2, 16, 16, 8),
            input_dtype="float32",
            expected_output_shape=(2, 16, 16, 8),
            expected_output_dtype="float32",
        )
        self.run_layer_test(
            SpectralDepthwiseConv,
            init_kwargs={
                "kernel_size": 3,
                "strides": 2,
                "padding": "same",
                "power_iterations": 1,
            },
            input_shape=(2, 16, 16, 8),
            input_dtype="float32",
            expected_output_shape=(2, 8, 8, 8),
            expected_output_dtype="float32",
        )

    def test_value(self):
        inputs = np.array(
            [
                0.298,
                -0.336,
                4.725,
                -0.393,
                -1.951,
                -3.269,
                -2.453,
                -0.043,
                0.25,
                -4.571,
                -2.143,
                2.744,
                1.483,
                3.229,
                1.472,
                -1.802,
                3.146,
                -0.048,
                1.407,
                1.315,
                -0.823,
                0.763,
                -0.103,
                0.295,
                1.507,
                -3.52,
                -1.55,
                -2.573,
                0.929,
                1.649,
                1.545,
                -0.365,
                1.845,
                1.208,
                -0.829,
                -3.652,
            ],
            "float32",
        ).reshape((1, 3, 4, 3))
        kernel = np.array(
            [
                0.307,
                -0.094,
                0.031,
                -0.301,
                -0.164,
                -0.073,
                0.07,
                -0.167,
                0.267,
                -0.128,
                -0.226,
                -0.181,
                -0.248,
                -0.05,
                0.056,
                -0.535,
                0.221,
                -0.04,
                0.521,
                -0.285,
                -0.323,
                0.094,
                0.362,
                -0.022,
                -0.097,
                -0.054,
                -0.084,
            ],
            "float32",
        ).reshape((3, 3, 3, 1))
        bias = np.zeros((3,), "float32")
        u = np.array([[-0.008, 0.009, -0.002]], "float32")
        expected = np.array(
            [
                -0.06966147,
                1.2172067,
                0.5698621,
                -1.6727608,
                -0.65479755,
                0.32279092,
            ],
            "float32",
        ).reshape((1, 1, 2, 3))

        layer = SpectralDepthwiseConv(3)
        layer.build(inputs.shape)
        layer.set_weights([kernel, bias, u])

        result = layer(inputs, training=True)
        self.assertAllClose(expected, result)

        result = layer(inputs)
        self.assertAllClose(expected, result)

    def test_batch(self):
        inputs = np.random.normal(size=(32, 16, 16, 64)) * 10.0
        layer = SpectralDepthwiseConv(3)

        expected = layer(inputs, training=True)

        result = layer(inputs, training=False)
        self.assertAllClose(expected, result)

        result = []
        for i in range(inputs.shape[0]):
            result1 = layer(inputs[i : i + 1], training=False)
            result.append(result1)
        result = np.concatenate(result, axis=0)
        self.assertAllClose(expected, result, atol=6e-5)
