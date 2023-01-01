import numpy as np
import tensorflow as tf
import unittest
from absl.testing import parameterized
from keras import layers
from keras.testing_infra import test_combinations, test_utils
from keras.mixed_precision import policy as mixed_precision
from tensorflow_addons import layers as add_layers
from segme.policy.norm import NORMALIZATIONS, BatchNorm, LayerNorm, TrueLayerNorm, GroupNorm, FilterResponseNorm


class TestNormalizationsRegistry(unittest.TestCase):
    def test_filled(self):
        self.assertIn('bn', NORMALIZATIONS)
        self.assertIn('gn', NORMALIZATIONS)


@test_combinations.run_all_keras_modes
class TestBatchNormalization(test_combinations.TestCase):
    def setUp(self):
        super(TestBatchNormalization, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestBatchNormalization, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            BatchNorm,
            kwargs={'data_format': 'channels_last'},
            input_shape=[2, 8, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 8, 16, 3],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            BatchNorm,
            kwargs={'data_format': 'channels_first'},
            input_shape=[2, 8, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 8, 16, 3],
            expected_output_dtype='float32'
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')

        result = test_utils.layer_test(
            BatchNorm,
            kwargs={'data_format': 'channels_last'},
            input_shape=[2, 8, 16, 3],
            input_dtype='float16',
            expected_output_shape=[None, 8, 16, 3],
            expected_output_dtype='float16'
        )
        self.assertTrue(np.all(np.isfinite(result)))

        result = test_utils.layer_test(
            BatchNorm,
            kwargs={'data_format': 'channels_first'},
            input_shape=[2, 8, 16, 3],
            input_dtype='float16',
            expected_output_shape=[None, 8, 16, 3],
            expected_output_dtype='float16'
        )
        self.assertTrue(np.all(np.isfinite(result)))

    @parameterized.parameters([
        ('channels_last', 'float32'), ('channels_last', 'float16'),
        ('channels_first', 'float32'), ('channels_first', 'float16')])
    def test_value(self, dformat, dtype):
        inputs = np.random.uniform(size=(2, 4, 4, 6)).astype('float32')

        if 'float16' == dtype:
            mixed_precision.set_global_policy('mixed_float16')
            inputs = inputs.astype(dtype)

        builtin = layers.BatchNormalization()
        expected = builtin(inputs)
        expected = self.evaluate(expected)

        if 'channels_first' == dformat:
            inputs = inputs.transpose(0, 3, 1, 2)
            expected = expected.transpose(0, 3, 1, 2)

        custom = BatchNorm(data_format=dformat)
        result = custom(inputs)
        result = self.evaluate(result)
        self.assertAllClose(expected, result)
        self.assertEqual(custom.fused, True)


@test_combinations.run_all_keras_modes
class TestLayerNorm(test_combinations.TestCase):
    def setUp(self):
        super(TestLayerNorm, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestLayerNorm, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            LayerNorm,
            kwargs={'data_format': 'channels_last'},
            input_shape=[2, 8, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 8, 16, 3],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            LayerNorm,
            kwargs={'data_format': 'channels_first'},
            input_shape=[2, 8, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 8, 16, 3],
            expected_output_dtype='float32'
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')

        result = test_utils.layer_test(
            LayerNorm,
            kwargs={'data_format': 'channels_last'},
            input_shape=[2, 8, 16, 3],
            input_dtype='float16',
            expected_output_shape=[None, 8, 16, 3],
            expected_output_dtype='float16'
        )
        self.assertTrue(np.all(np.isfinite(result)))

        result = test_utils.layer_test(
            LayerNorm,
            kwargs={'data_format': 'channels_first'},
            input_shape=[2, 8, 16, 3],
            input_dtype='float16',
            expected_output_shape=[None, 8, 16, 3],
            expected_output_dtype='float16'
        )
        self.assertTrue(np.all(np.isfinite(result)))

    @parameterized.parameters([
        ('channels_last', 'float32'), ('channels_last', 'float16'),
        ('channels_first', 'float32'), ('channels_first', 'float16')])
    def test_value(self, dformat, dtype):
        inputs = np.random.uniform(size=(2, 4, 4, 6)).astype('float32')

        if 'float16' == dtype:
            mixed_precision.set_global_policy('mixed_float16')
            inputs = inputs.astype(dtype)

        builtin = layers.LayerNormalization()
        expected = builtin(inputs)
        expected = self.evaluate(expected)

        if 'channels_first' == dformat:
            inputs = inputs.transpose(0, 3, 1, 2)
            expected = expected.transpose(0, 3, 1, 2)

        custom = LayerNorm(data_format=dformat)
        result = custom(inputs)
        result = self.evaluate(result)
        self.assertAllClose(expected, result)
        self.assertEqual(custom._fused, 'channels_last' == dformat)

    def test_batch(self):
        inputs = np.random.normal(size=(32, 16, 16, 64)) * 10.
        layer = LayerNorm()

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
        self.assertAllClose(expected, result)


@test_combinations.run_all_keras_modes
class TestTrueLayerNorm(test_combinations.TestCase):
    def setUp(self):
        super(TestTrueLayerNorm, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestTrueLayerNorm, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            TrueLayerNorm,
            kwargs={'data_format': 'channels_last'},
            input_shape=[2, 8, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 8, 16, 3],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            TrueLayerNorm,
            kwargs={'data_format': 'channels_first'},
            input_shape=[2, 8, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 8, 16, 3],
            expected_output_dtype='float32'
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')

        result = test_utils.layer_test(
            TrueLayerNorm,
            kwargs={'data_format': 'channels_last'},
            input_shape=[2, 8, 16, 3],
            input_dtype='float16',
            expected_output_shape=[None, 8, 16, 3],
            expected_output_dtype='float16'
        )
        self.assertTrue(np.all(np.isfinite(result)))

        result = test_utils.layer_test(
            TrueLayerNorm,
            kwargs={'data_format': 'channels_first'},
            input_shape=[2, 8, 16, 3],
            input_dtype='float16',
            expected_output_shape=[None, 8, 16, 3],
            expected_output_dtype='float16'
        )
        self.assertTrue(np.all(np.isfinite(result)))

    @parameterized.parameters([
        ('channels_last', 'float32'), ('channels_last', 'float16'),
        ('channels_first', 'float32'), ('channels_first', 'float16')])
    def test_value(self, dformat, dtype):
        inputs = np.random.uniform(size=(2, 4, 4, 6)).astype('float32')

        if 'float16' == dtype:
            mixed_precision.set_global_policy('mixed_float16')
            inputs = inputs.astype(dtype)

        builtin = layers.LayerNormalization(axis=[1, 2, 3])
        expected = builtin(inputs)
        expected = self.evaluate(expected)

        if 'channels_first' == dformat:
            inputs = inputs.transpose(0, 3, 1, 2)
            expected = expected.transpose(0, 3, 1, 2)

        custom = TrueLayerNorm(data_format=dformat)
        result = custom(inputs)
        result = self.evaluate(result)
        self.assertAllClose(expected, result)
        self.assertEqual(custom._fused, True)

    def test_batch(self):
        inputs = np.random.normal(size=(32, 16, 16, 64)) * 10.
        layer = TrueLayerNorm()

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
        self.assertAllClose(expected, result, atol=1.1e-5)


@test_combinations.run_all_keras_modes
class TestGroupNorm(test_combinations.TestCase):
    def setUp(self):
        super(TestGroupNorm, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestGroupNorm, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            GroupNorm,
            kwargs={'data_format': 'channels_last'},
            input_shape=[2, 8, 16, 64],
            input_dtype='float32',
            expected_output_shape=[None, 8, 16, 64],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            GroupNorm,
            kwargs={'data_format': 'channels_first'},
            input_shape=[2, 8, 16, 64],
            input_dtype='float32',
            expected_output_shape=[None, 8, 16, 64],
            expected_output_dtype='float32'
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        result = test_utils.layer_test(
            GroupNorm,
            kwargs={'data_format': 'channels_last'},
            input_shape=[2, 8, 16, 64],
            input_dtype='float16',
            expected_output_shape=[None, 8, 16, 64],
            expected_output_dtype='float16'
        )
        self.assertTrue(np.all(np.isfinite(result)))

        result = test_utils.layer_test(
            GroupNorm,
            kwargs={'data_format': 'channels_first'},
            input_shape=[2, 8, 16, 64],
            input_dtype='float16',
            expected_output_shape=[None, 8, 16, 64],
            expected_output_dtype='float16'
        )
        self.assertTrue(np.all(np.isfinite(result)))

    @parameterized.parameters([
        ('channels_last', 'float32'), ('channels_last', 'float16'),
        ('channels_first', 'float32'), ('channels_first', 'float16')])
    def test_value(self, dformat, dtype):
        inputs = np.random.uniform(size=(2, 4, 4, 6)).astype('float32')

        if 'float16' == dtype:
            mixed_precision.set_global_policy('mixed_float16')
            inputs = inputs.astype(dtype)

        builtin = layers.GroupNormalization(2)
        expected = builtin(inputs)
        expected = self.evaluate(expected)

        if 'channels_first' == dformat:
            inputs = inputs.transpose(0, 3, 1, 2)
            expected = expected.transpose(0, 3, 1, 2)

        custom = GroupNorm(data_format=dformat)
        result = custom(inputs)
        result = self.evaluate(result)
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

        layer.build([2, 4, 4, 2])
        self.assertEqual(layer.groups, 1)

        layer.build([2, 4, 4, 6])
        self.assertEqual(layer.groups, 2)

        layer.build([2, 4, 4, 8])
        self.assertEqual(layer.groups, 2)

        layer.build([2, 4, 4, 16])
        self.assertEqual(layer.groups, 4)

        layer.build([2, 4, 4, 32])
        self.assertEqual(layer.groups, 4)

        layer.build([2, 4, 4, 64])
        self.assertEqual(layer.groups, 8)

        layer.build([2, 4, 4, 128])
        self.assertEqual(layer.groups, 8)

        layer.build([2, 4, 4, 256])
        self.assertEqual(layer.groups, 16)

        layer.build([2, 4, 4, 512])
        self.assertEqual(layer.groups, 32)

        layer.build([2, 4, 4, 1024])
        self.assertEqual(layer.groups, 32)

        layer.build([2, 4, 4, 2048])
        self.assertEqual(layer.groups, 32)

    def test_batch(self):
        inputs = np.random.normal(size=(32, 16, 16, 64)) * 10.
        layer = GroupNorm()

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
        self.assertAllClose(expected, result)


@test_combinations.run_all_keras_modes
class TestFilterResponseNorm(test_combinations.TestCase):
    def setUp(self):
        super(TestFilterResponseNorm, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestFilterResponseNorm, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            FilterResponseNorm,
            kwargs={'data_format': 'channels_last'},
            input_shape=[2, 8, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 8, 16, 3],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            FilterResponseNorm,
            kwargs={'data_format': 'channels_first'},
            input_shape=[2, 8, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 8, 16, 3],
            expected_output_dtype='float32'
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')

        result = test_utils.layer_test(
            FilterResponseNorm,
            kwargs={'data_format': 'channels_last'},
            input_shape=[2, 8, 16, 3],
            input_dtype='float16',
            expected_output_shape=[None, 8, 16, 3],
            expected_output_dtype='float16'
        )
        self.assertTrue(np.all(np.isfinite(result)))

        result = test_utils.layer_test(
            FilterResponseNorm,
            kwargs={'data_format': 'channels_first'},
            input_shape=[2, 8, 16, 3],
            input_dtype='float16',
            expected_output_shape=[None, 8, 16, 3],
            expected_output_dtype='float16'
        )
        self.assertTrue(np.all(np.isfinite(result)))

    @parameterized.parameters(
        [('channels_last', 'float32'), ('channels_last', 'float16'),
         ('channels_first', 'float32'), ('channels_first', 'float16')])
    def test_value(self, dformat, dtype):
        inputs = np.random.uniform(size=(2, 4, 4, 6)).astype('float32')

        builtin = add_layers.FilterResponseNormalization()
        expected = builtin(inputs)
        expected = self.evaluate(expected)

        if 'float16' == dtype:
            mixed_precision.set_global_policy('mixed_float16')
            inputs = inputs.astype(dtype)
            expected = expected.astype(dtype)

        if 'channels_first' == dformat:
            inputs = inputs.transpose(0, 3, 1, 2)
            expected = expected.transpose(0, 3, 1, 2)

        custom = FilterResponseNorm(data_format=dformat)
        result = custom(inputs)
        result = self.evaluate(result)
        self.assertAllClose(expected, result, atol=1e-3)

    def test_batch(self):
        inputs = np.random.normal(size=(32, 16, 16, 64)) * 10.
        layer = FilterResponseNorm()

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
        self.assertAllClose(expected, result)


if __name__ == '__main__':
    tf.test.main()
