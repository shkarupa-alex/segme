import numpy as np
import tensorflow as tf
import unittest
from keras.testing_infra import test_combinations, test_utils
from keras.mixed_precision import policy as mixed_precision
from segme.policy.sameconv import SAMECONVS, SameConv, SameStandardizedConv, SameSpectralConv, SameDepthwiseConv


class TestSamConvsRegistry(unittest.TestCase):
    def test_filled(self):
        self.assertIn('conv', SAMECONVS)
        self.assertIn('stdconv', SAMECONVS)


@test_combinations.run_all_keras_modes
class TestSameConv(test_combinations.TestCase):
    def setUp(self):
        super(TestSameConv, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestSameConv, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            SameConv,
            kwargs={'filters': 4, 'kernel_size': 1, 'strides': 1},
            input_shape=[2, 16, 16, 8],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 4],
            expected_output_dtype='float32'
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        result = test_utils.layer_test(
            SameConv,
            kwargs={'filters': 4, 'kernel_size': 3, 'strides': 2},
            input_shape=[2, 16, 16, 8],
            input_dtype='float16',
            expected_output_shape=[None, 8, 8, 4],
            expected_output_dtype='float16'
        )
        self.assertTrue(np.all(np.isfinite(result)))


@test_combinations.run_all_keras_modes
class TestSameStandardizedConv(test_combinations.TestCase):
    def setUp(self):
        super(TestSameStandardizedConv, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestSameStandardizedConv, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            SameStandardizedConv,
            kwargs={'filters': 4, 'kernel_size': 1, 'strides': 1},
            input_shape=[2, 16, 16, 8],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 4],
            expected_output_dtype='float32'
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        result = test_utils.layer_test(
            SameStandardizedConv,
            kwargs={'filters': 4, 'kernel_size': 3, 'strides': 2},
            input_shape=[2, 16, 16, 8],
            input_dtype='float16',
            expected_output_shape=[None, 8, 8, 4],
            expected_output_dtype='float16'
        )
        self.assertTrue(np.all(np.isfinite(result)))


@test_combinations.run_all_keras_modes
class TestSameSpectralConv(test_combinations.TestCase):
    def setUp(self):
        super(TestSameSpectralConv, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestSameSpectralConv, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            SameSpectralConv,
            kwargs={'filters': 4, 'kernel_size': 1, 'strides': 1},
            input_shape=[2, 16, 16, 8],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 4],
            expected_output_dtype='float32'
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        result = test_utils.layer_test(
            SameSpectralConv,
            kwargs={'filters': 4, 'kernel_size': 3, 'strides': 2},
            input_shape=[2, 16, 16, 8],
            input_dtype='float16',
            expected_output_shape=[None, 8, 8, 4],
            expected_output_dtype='float16'
        )
        self.assertTrue(np.all(np.isfinite(result)))


@test_combinations.run_all_keras_modes
class TestSameDepthwiseConv(test_combinations.TestCase):
    def setUp(self):
        super(TestSameDepthwiseConv, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestSameDepthwiseConv, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            SameDepthwiseConv,
            kwargs={'kernel_size': 1, 'strides': 1},
            input_shape=[2, 16, 16, 8],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 8],
            expected_output_dtype='float32'
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        result = test_utils.layer_test(
            SameDepthwiseConv,
            kwargs={'kernel_size': 3, 'strides': 2},
            input_shape=[2, 16, 16, 8],
            input_dtype='float16',
            expected_output_shape=[None, 8, 8, 8],
            expected_output_dtype='float16'
        )
        self.assertTrue(np.all(np.isfinite(result)))


if __name__ == '__main__':
    tf.test.main()
