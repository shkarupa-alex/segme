import numpy as np
import tensorflow as tf
import unittest
from keras.testing_infra import test_combinations, test_utils
from keras.mixed_precision import policy as mixed_precision
from segme.policy.norm import NORMALIZATIONS, LayerNormalization, GroupNormalization, FilterResponseNormalization


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
            NORMALIZATIONS['bn'],
            kwargs={},
            input_shape=[2, 8, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 8, 16, 3],
            expected_output_dtype='float32'
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        result = test_utils.layer_test(
            NORMALIZATIONS['bn'],
            kwargs={},
            input_shape=[2, 8, 16, 3],
            input_dtype='float16',
            expected_output_shape=[None, 8, 16, 3],
            expected_output_dtype='float16'
        )
        self.assertTrue(np.all(np.isfinite(result)))


@test_combinations.run_all_keras_modes
class TestLayerNormalization(test_combinations.TestCase):
    def setUp(self):
        super(TestLayerNormalization, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestLayerNormalization, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            LayerNormalization,
            kwargs={},
            input_shape=[2, 8, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 8, 16, 3],
            expected_output_dtype='float32'
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        result = test_utils.layer_test(
            LayerNormalization,
            kwargs={},
            input_shape=[2, 8, 16, 3],
            input_dtype='float16',
            expected_output_shape=[None, 8, 16, 3],
            expected_output_dtype='float16'
        )
        self.assertTrue(np.all(np.isfinite(result)))


@test_combinations.run_all_keras_modes
class TestGroupNormalization(test_combinations.TestCase):
    def setUp(self):
        super(TestGroupNormalization, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestGroupNormalization, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            GroupNormalization,
            kwargs={},
            input_shape=[2, 8, 16, 64],
            input_dtype='float32',
            expected_output_shape=[None, 8, 16, 64],
            expected_output_dtype='float32'
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        result = test_utils.layer_test(
            GroupNormalization,
            kwargs={},
            input_shape=[2, 8, 16, 3],
            input_dtype='float16',
            expected_output_shape=[None, 8, 16, 3],
            expected_output_dtype='float16'
        )
        self.assertTrue(np.all(np.isfinite(result)))


@test_combinations.run_all_keras_modes
class TestFilterResponseNormalization(test_combinations.TestCase):
    def setUp(self):
        super(TestFilterResponseNormalization, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestFilterResponseNormalization, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            FilterResponseNormalization,
            kwargs={},
            input_shape=[2, 8, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 8, 16, 3],
            expected_output_dtype='float32'
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        result = test_utils.layer_test(
            FilterResponseNormalization,
            kwargs={},
            input_shape=[2, 8, 16, 3],
            input_dtype='float16',
            expected_output_shape=[None, 8, 16, 3],
            expected_output_dtype='float16'
        )
        self.assertTrue(np.all(np.isfinite(result)))


if __name__ == '__main__':
    tf.test.main()
