import numpy as np
import tensorflow as tf
import unittest
from keras.testing_infra import test_combinations, test_utils
from keras.mixed_precision import policy as mixed_precision
from segme.policy.act import ACTIVATIONS, TLU


class TestActivationsRegistry(unittest.TestCase):
    def test_filled(self):
        self.assertIn('relu', ACTIVATIONS)
        self.assertIn('leakyrelu', ACTIVATIONS)


@test_combinations.run_all_keras_modes
class TestTLU(test_combinations.TestCase):
    def setUp(self):
        super(TestTLU, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestTLU, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            TLU,
            kwargs={},
            input_shape=[2, 8, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 8, 16, 3],
            expected_output_dtype='float32'
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        result = test_utils.layer_test(
            TLU,
            kwargs={},
            input_shape=[2, 8, 16, 3],
            input_dtype='float16',
            expected_output_shape=[None, 8, 16, 3],
            expected_output_dtype='float16'
        )
        self.assertTrue(np.all(np.isfinite(result)))


if __name__ == '__main__':
    tf.test.main()
