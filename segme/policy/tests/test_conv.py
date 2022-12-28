import itertools
import numpy as np
import tensorflow as tf
import unittest
from keras import layers
from keras.testing_infra import test_combinations, test_utils
from keras.mixed_precision import policy as mixed_precision
from segme.policy.conv import CONVOLUTIONS, FixedConv, FixedDepthwiseConv, StandardizedConv, \
    StandardizedDepthwiseConv, SpectralConv, SpectralDepthwiseConv


class TestConvsRegistry(unittest.TestCase):
    def test_filled(self):
        self.assertIn('conv', CONVOLUTIONS)
        self.assertIn('stdconv', CONVOLUTIONS)


@test_combinations.run_all_keras_modes
class TestFixedConv(test_combinations.TestCase):
    def setUp(self):
        super(TestFixedConv, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestFixedConv, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            FixedConv,
            kwargs={'filters': 4, 'kernel_size': 3, 'strides': 1, 'dilation_rate': 3, 'padding': 'valid'},
            input_shape=[2, 16, 16, 8],
            input_dtype='float32',
            expected_output_shape=[None, 10, 10, 4],
            expected_output_dtype='float32'
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        result = test_utils.layer_test(
            FixedConv,
            kwargs={'filters': 4, 'kernel_size': 3, 'strides': 2, 'dilation_rate': 1, 'padding': 'same'},
            input_shape=[2, 16, 16, 8],
            input_dtype='float16',
            expected_output_shape=[None, 8, 8, 4],
            expected_output_dtype='float16'
        )
        self.assertTrue(np.all(np.isfinite(result)))

    def test_valid(self):
        for k, s, d in itertools.product(range(1, 9), range(1, 9), range(1, 9)):
            if s > 1 and d > 1:
                continue

            exconv = layers.Conv2D(2, k, strides=s, dilation_rate=d, padding='valid')
            exconv.build([None, None, None, 4])

            layer = FixedConv(2, k, strides=s, dilation_rate=d, padding='valid')
            layer.build([None, None, None, 4])
            layer.set_weights(exconv.get_weights())

            for h, w in itertools.product(range(128, 128 + 16 + 1), range(128, 128 + 16 + 1)):
                inputs = np.random.normal(size=(2, h, w, 4)) * 10.

                exshape = exconv.compute_output_shape(inputs.shape)
                exshape = tuple(exshape.as_list())

                exval = exconv(inputs)
                exval = self.evaluate(exval)

                result = layer(inputs)
                result = self.evaluate(result)

                self.assertTupleEqual(exshape, result.shape)
                self.assertLess(np.abs(exval - result).max(), 1e-4)

    def test_same(self):
        for k, s, d in itertools.product(range(1, 9), range(1, 9), range(1, 9)):
            if s > 1 and d > 1:
                continue

            exshapeconv = layers.Conv2D(2, k, strides=s, dilation_rate=d, padding='same')

            exvalconv = layers.Conv2D(2, k, strides=s, dilation_rate=d, padding='valid')
            exvalconv.build([None, None, None, 4])

            layer = FixedConv(2, k, strides=s, dilation_rate=d, padding='same')
            layer.build([None, None, None, 4])
            layer.set_weights(exvalconv.get_weights())

            for h, w in itertools.product(range(128, 128 + 16 + 1), range(128, 128 + 16 + 1)):
                inputs = np.random.normal(size=(2, h, w, 4)) * 10.

                paddings = d * (k - 1)
                paddings = (paddings // 2, paddings - paddings // 2)
                painputs = np.pad(inputs, ((0, 0), paddings, paddings, (0, 0)))

                exshape = exshapeconv.compute_output_shape(inputs.shape)
                exshape = tuple(exshape.as_list())

                exval = exvalconv(painputs)
                exval = self.evaluate(exval)

                result = layer(inputs)
                result = self.evaluate(result)

                self.assertTupleEqual(exshape, result.shape)
                self.assertLess(np.abs(exval - result).max(), 1e-4)


@test_combinations.run_all_keras_modes
class TestFixedDepthwiseConv(test_combinations.TestCase):
    def setUp(self):
        super(TestFixedDepthwiseConv, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestFixedDepthwiseConv, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            FixedDepthwiseConv,
            kwargs={'kernel_size': 3, 'strides': 1, 'dilation_rate': 3, 'padding': 'same'},
            input_shape=[2, 16, 16, 8],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 8],
            expected_output_dtype='float32'
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        result = test_utils.layer_test(
            FixedDepthwiseConv,
            kwargs={'kernel_size': 3, 'strides': 2, 'dilation_rate': 1, 'padding': 'valid'},
            input_shape=[2, 16, 16, 8],
            input_dtype='float16',
            expected_output_shape=[None, 7, 7, 8],
            expected_output_dtype='float16'
        )
        self.assertTrue(np.all(np.isfinite(result)))

    def test_valid(self):
        for k, s, d in itertools.product(range(1, 9), range(1, 9), range(1, 9)):
            if s > 1 and d > 1:
                continue

            exconv = layers.DepthwiseConv2D(k, strides=s, dilation_rate=d, padding='valid')
            exconv.build([None, None, None, 4])

            layer = FixedDepthwiseConv(k, strides=s, dilation_rate=d, padding='valid')
            layer.build([None, None, None, 4])
            layer.set_weights(exconv.get_weights())

            for h, w in itertools.product(range(128, 128 + 16 + 1), range(128, 128 + 16 + 1)):
                inputs = np.random.normal(size=(2, h, w, 4)) * 10.

                exshape = exconv.compute_output_shape(inputs.shape)
                exshape = tuple(exshape.as_list())

                exval = exconv(inputs)
                exval = self.evaluate(exval)

                result = layer(inputs)
                result = self.evaluate(result)

                self.assertTupleEqual(exshape, result.shape)
                self.assertLess(np.abs(exval - result).max(), 1e-4)

    def test_same(self):
        for k, s, d in itertools.product(range(1, 9), range(1, 9), range(1, 9)):
            if s > 1 and d > 1:
                continue

            exshapeconv = layers.DepthwiseConv2D(k, strides=s, dilation_rate=d, padding='same')

            exvalconv = layers.DepthwiseConv2D(k, strides=s, dilation_rate=d, padding='valid')
            exvalconv.build([None, None, None, 4])

            layer = FixedDepthwiseConv(k, strides=s, dilation_rate=d, padding='same')
            layer.build([None, None, None, 4])
            layer.set_weights(exvalconv.get_weights())

            for h, w in itertools.product(range(128, 128 + 16 + 1), range(128, 128 + 16 + 1)):
                inputs = np.random.normal(size=(2, h, w, 4)) * 10.

                paddings = d * (k - 1)
                paddings = (paddings // 2, paddings - paddings // 2)
                painputs = np.pad(inputs, ((0, 0), paddings, paddings, (0, 0)))

                exshape = exshapeconv.compute_output_shape(inputs.shape)
                exshape = tuple(exshape.as_list())

                exval = exvalconv(painputs)
                exval = self.evaluate(exval)

                result = layer(inputs)
                result = self.evaluate(result)

                self.assertTupleEqual(exshape, result.shape)
                self.assertLess(np.abs(exval - result).max(), 1e-4)


@test_combinations.run_all_keras_modes
class TestStandardizedConv(test_combinations.TestCase):
    def setUp(self):
        super(TestStandardizedConv, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestStandardizedConv, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            StandardizedConv,
            kwargs={'filters': 4, 'kernel_size': 1, 'strides': 1, 'padding': 'valid'},
            input_shape=[2, 16, 16, 8],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 4],
            expected_output_dtype='float32'
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        result = test_utils.layer_test(
            StandardizedConv,
            kwargs={'filters': 4, 'kernel_size': 3, 'strides': 2, 'padding': 'same'},
            input_shape=[2, 16, 16, 8],
            input_dtype='float16',
            expected_output_shape=[None, 8, 8, 4],
            expected_output_dtype='float16'
        )
        self.assertTrue(np.all(np.isfinite(result)))

    def test_value(self):
        inputs = np.array([
            0.298, -0.336, 4.725, -0.393, -1.951, -3.269, -2.453, -0.043, 0.25, -4.571, -2.143, 2.744, 1.483, 3.229,
            1.472, -1.802, 3.146, -0.048, 1.407, 1.315, -0.823, 0.763, -0.103, 0.295, 1.507, -3.52, -1.55, -2.573,
            0.929, 1.649, 1.545, -0.365, 1.845, 1.208, -0.829, -3.652], 'float32').reshape((1, 3, 4, 3))
        kernel = np.array([
            0.307, -0.094, 0.031, -0.301, -0.164, -0.073, 0.07, -0.167, 0.267, -0.128, -0.226, -0.181, -0.248, -0.05,
            0.056, -0.535, 0.221, -0.04, 0.521, -0.285, -0.323, 0.094, 0.362, -0.022, -0.097, -0.054, -0.084],
            'float32').reshape((3, 3, 3, 1))
        bias = np.zeros((1,), 'float32')
        expected = np.array([7.993624, -9.57225], 'float32').reshape((1, 1, 2, 1))

        layer = StandardizedConv(1, 3)
        layer.build(inputs.shape)
        layer.set_weights([kernel, bias])

        result = layer(inputs)
        result = self.evaluate(result)
        self.assertAllClose(expected, result)

    def test_value_nonfused(self):
        mixed_precision.set_global_policy('float16')
        inputs = np.array([
            0.298, -0.336, 4.725, -0.393, -1.951, -3.269, -2.453, -0.043, 0.25, -4.571, -2.143, 2.744, 1.483, 3.229,
            1.472, -1.802, 3.146, -0.048, 1.407, 1.315, -0.823, 0.763, -0.103, 0.295, 1.507, -3.52, -1.55, -2.573,
            0.929, 1.649, 1.545, -0.365, 1.845, 1.208, -0.829, -3.652], 'float32').reshape((1, 3, 4, 3))
        kernel = np.array([
            0.307, -0.094, 0.031, -0.301, -0.164, -0.073, 0.07, -0.167, 0.267, -0.128, -0.226, -0.181, -0.248, -0.05,
            0.056, -0.535, 0.221, -0.04, 0.521, -0.285, -0.323, 0.094, 0.362, -0.022, -0.097, -0.054, -0.084],
            'float32').reshape((3, 3, 3, 1))
        bias = np.zeros((1,), 'float32')
        expected = np.array([7.993624, -9.57225], 'float32').reshape((1, 1, 2, 1))

        layer = StandardizedConv(1, 3)
        layer.build(inputs.shape)
        layer.set_weights([kernel, bias])

        result = layer(inputs)
        result = self.evaluate(result)
        self.assertAllClose(expected, result, atol=7e-3)

    def test_batch(self):
        inputs = np.random.normal(size=(32, 16, 16, 64)) * 10.
        layer = StandardizedConv(64, 3)

        expected = layer(inputs, training=True)
        expected = self.evaluate(expected)

        result = layer(inputs, training=False)
        result = self.evaluate(result)
        self.assertAllClose(expected, result)

        result = []
        for i in range(inputs.shape[0]):
            result1 = layer(inputs[i:i + 1], training=False)
            result1 = self.evaluate(result1)
            result.append(result1)
        result = np.concatenate(result, axis=0)
        self.assertAllClose(expected, result, atol=3e-4)


@test_combinations.run_all_keras_modes
class TestStandardizedDepthwiseConv(test_combinations.TestCase):
    def setUp(self):
        super(TestStandardizedDepthwiseConv, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestStandardizedDepthwiseConv, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            StandardizedDepthwiseConv,
            kwargs={'kernel_size': 1, 'strides': 1, 'padding': 'valid'},
            input_shape=[2, 16, 16, 8],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 8],
            expected_output_dtype='float32'
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        result = test_utils.layer_test(
            StandardizedDepthwiseConv,
            kwargs={'kernel_size': 3, 'strides': 2, 'padding': 'same'},
            input_shape=[2, 16, 16, 8],
            input_dtype='float16',
            expected_output_shape=[None, 8, 8, 8],
            expected_output_dtype='float16'
        )
        self.assertTrue(np.all(np.isfinite(result)))

    def test_value(self):
        inputs = np.array([
            0.298, -0.336, 4.725, -0.393, -1.951, -3.269, -2.453, -0.043, 0.25, -4.571, -2.143, 2.744, 1.483, 3.229,
            1.472, -1.802, 3.146, -0.048, 1.407, 1.315, -0.823, 0.763, -0.103, 0.295, 1.507, -3.52, -1.55, -2.573,
            0.929, 1.649, 1.545, -0.365, 1.845, 1.208, -0.829, -3.652], 'float32').reshape((1, 3, 4, 3))
        kernel = np.array([
            0.307, -0.094, 0.031, -0.301, -0.164, -0.073, 0.07, -0.167, 0.267, -0.128, -0.226, -0.181, -0.248, -0.05,
            0.056, -0.535, 0.221, -0.04, 0.521, -0.285, -0.323, 0.094, 0.362, -0.022, -0.097, -0.054, -0.084],
            'float32').reshape((3, 3, 3, 1))
        bias = np.zeros((3,), 'float32')
        expected = np.array(
            [-0.32465044, 6.2342463, 4.557652, -5.863132, -3.0357158, 1.6686258], 'float32').reshape((1, 1, 2, 3))

        layer = StandardizedDepthwiseConv(3)
        layer.build(inputs.shape)
        layer.set_weights([kernel, bias])

        result = layer(inputs)
        result = self.evaluate(result)
        self.assertAllClose(expected, result)

    def test_value_nonfused(self):
        mixed_precision.set_global_policy('float16')
        inputs = np.array([
            0.298, -0.336, 4.725, -0.393, -1.951, -3.269, -2.453, -0.043, 0.25, -4.571, -2.143, 2.744, 1.483, 3.229,
            1.472, -1.802, 3.146, -0.048, 1.407, 1.315, -0.823, 0.763, -0.103, 0.295, 1.507, -3.52, -1.55, -2.573,
            0.929, 1.649, 1.545, -0.365, 1.845, 1.208, -0.829, -3.652], 'float32').reshape((1, 3, 4, 3))
        kernel = np.array([
            0.307, -0.094, 0.031, -0.301, -0.164, -0.073, 0.07, -0.167, 0.267, -0.128, -0.226, -0.181, -0.248, -0.05,
            0.056, -0.535, 0.221, -0.04, 0.521, -0.285, -0.323, 0.094, 0.362, -0.022, -0.097, -0.054, -0.084],
            'float32').reshape((3, 3, 3, 1))
        bias = np.zeros((3,), 'float32')
        expected = np.array(
            [-0.32465044, 6.2342463, 4.557652, -5.863132, -3.0357158, 1.6686258], 'float32').reshape((1, 1, 2, 3))

        layer = StandardizedDepthwiseConv(3)
        layer.build(inputs.shape)
        layer.set_weights([kernel, bias])

        result = layer(inputs)
        result = self.evaluate(result)
        self.assertAllClose(expected, result, atol=7e-3)

    def test_batch(self):
        inputs = np.random.normal(size=(32, 16, 16, 64)) * 10.
        layer = StandardizedDepthwiseConv(3)

        expected = layer(inputs, training=True)
        expected = self.evaluate(expected)

        result = layer(inputs, training=False)
        result = self.evaluate(result)
        self.assertAllClose(expected, result)

        result = []
        for i in range(inputs.shape[0]):
            result1 = layer(inputs[i:i + 1], training=False)
            result1 = self.evaluate(result1)
            result.append(result1)
        result = np.concatenate(result, axis=0)
        self.assertAllClose(expected, result, atol=1e-4)


@test_combinations.run_all_keras_modes
class TestSpectralConv(test_combinations.TestCase):
    def setUp(self):
        super(TestSpectralConv, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestSpectralConv, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            SpectralConv,
            kwargs={'filters': 4, 'kernel_size': 1, 'strides': 1, 'padding': 'valid', 'power_iterations': 2},
            input_shape=[2, 16, 16, 8],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 4],
            expected_output_dtype='float32'
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        result = test_utils.layer_test(
            SpectralConv,
            kwargs={'filters': 4, 'kernel_size': 3, 'strides': 2, 'padding': 'same', 'power_iterations': 1},
            input_shape=[2, 16, 16, 8],
            input_dtype='float16',
            expected_output_shape=[None, 8, 8, 4],
            expected_output_dtype='float16'
        )
        self.assertTrue(np.all(np.isfinite(result)))

    def test_value(self):
        inputs = np.array([
            0.298, -0.336, 4.725, -0.393, -1.951, -3.269, -2.453, -0.043, 0.25, -4.571, -2.143, 2.744, 1.483, 3.229,
            1.472, -1.802, 3.146, -0.048, 1.407, 1.315, -0.823, 0.763, -0.103, 0.295, 1.507, -3.52, -1.55, -2.573,
            0.929, 1.649, 1.545, -0.365, 1.845, 1.208, -0.829, -3.652], 'float32').reshape((1, 3, 4, 3))
        kernel = np.array([
            0.307, -0.094, 0.031, -0.301, -0.164, -0.073, 0.07, -0.167, 0.267, -0.128, -0.226, -0.181, -0.248, -0.05,
            0.056, -0.535, 0.221, -0.04, 0.521, -0.285, -0.323, 0.094, 0.362, -0.022, -0.097, -0.054, -0.084],
            'float32').reshape((3, 3, 3, 1))
        bias = np.zeros((1,), 'float32')
        u = np.array([[-0.008]], 'float32')
        expected = np.array([1.3133075, -1.533053], 'float32').reshape((1, 1, 2, 1))

        layer = SpectralConv(1, 3)
        layer.build(inputs.shape)
        layer.set_weights([kernel, bias, u])

        result = layer(inputs, training=True)
        result = self.evaluate(result)
        self.assertAllClose(expected, result)

        result = layer(inputs)
        result = self.evaluate(result)
        self.assertAllClose(expected, result)

    def test_batch(self):
        inputs = np.random.normal(size=(32, 16, 16, 64)) * 10.
        layer = SpectralConv(64, 3)

        expected = layer(inputs, training=True)
        expected = self.evaluate(expected)

        result = layer(inputs, training=False)
        result = self.evaluate(result)
        self.assertAllClose(expected, result)

        result = []
        for i in range(inputs.shape[0]):
            result1 = layer(inputs[i:i + 1], training=False)
            result1 = self.evaluate(result1)
            result.append(result1)
        result = np.concatenate(result, axis=0)
        self.assertAllClose(expected, result, atol=6e-5)


@test_combinations.run_all_keras_modes
class TestSpectralDepthwiseConv(test_combinations.TestCase):
    def setUp(self):
        super(TestSpectralDepthwiseConv, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestSpectralDepthwiseConv, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            SpectralDepthwiseConv,
            kwargs={'kernel_size': 1, 'strides': 1, 'padding': 'valid', 'power_iterations': 2},
            input_shape=[2, 16, 16, 8],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 8],
            expected_output_dtype='float32'
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        result = test_utils.layer_test(
            SpectralDepthwiseConv,
            kwargs={'kernel_size': 3, 'strides': 2, 'padding': 'same', 'power_iterations': 1},
            input_shape=[2, 16, 16, 8],
            input_dtype='float16',
            expected_output_shape=[None, 8, 8, 8],
            expected_output_dtype='float16'
        )
        self.assertTrue(np.all(np.isfinite(result)))

    def test_value(self):
        inputs = np.array([
            0.298, -0.336, 4.725, -0.393, -1.951, -3.269, -2.453, -0.043, 0.25, -4.571, -2.143, 2.744, 1.483, 3.229,
            1.472, -1.802, 3.146, -0.048, 1.407, 1.315, -0.823, 0.763, -0.103, 0.295, 1.507, -3.52, -1.55, -2.573,
            0.929, 1.649, 1.545, -0.365, 1.845, 1.208, -0.829, -3.652], 'float32').reshape((1, 3, 4, 3))
        kernel = np.array([
            0.307, -0.094, 0.031, -0.301, -0.164, -0.073, 0.07, -0.167, 0.267, -0.128, -0.226, -0.181, -0.248, -0.05,
            0.056, -0.535, 0.221, -0.04, 0.521, -0.285, -0.323, 0.094, 0.362, -0.022, -0.097, -0.054, -0.084],
            'float32').reshape((3, 3, 3, 1))
        bias = np.zeros((3,), 'float32')
        u = np.array([[-0.008, 0.009, -0.002]], 'float32')
        expected = np.array([
            -0.06966147, 1.2172067, 0.5698621, -1.6727608, -0.65479755, 0.32279092], 'float32').reshape((1, 1, 2, 3))

        layer = SpectralDepthwiseConv(3)
        layer.build(inputs.shape)
        layer.set_weights([kernel, bias, u])

        result = layer(inputs, training=True)
        result = self.evaluate(result)
        self.assertAllClose(expected, result)

        result = layer(inputs)
        result = self.evaluate(result)
        self.assertAllClose(expected, result)

    def test_batch(self):
        inputs = np.random.normal(size=(32, 16, 16, 64)) * 10.
        layer = SpectralDepthwiseConv(3)

        expected = layer(inputs, training=True)
        expected = self.evaluate(expected)

        result = layer(inputs, training=False)
        result = self.evaluate(result)
        self.assertAllClose(expected, result)

        result = []
        for i in range(inputs.shape[0]):
            result1 = layer(inputs[i:i + 1], training=False)
            result1 = self.evaluate(result1)
            result.append(result1)
        result = np.concatenate(result, axis=0)
        self.assertAllClose(expected, result, atol=6e-5)


if __name__ == '__main__':
    tf.test.main()
