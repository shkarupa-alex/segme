import numpy as np
import tensorflow as tf
from keras.testing_infra import test_combinations, test_utils
from keras.mixed_precision import policy as mixed_precision
from segme.common.internear import NearestInterpolation
from segme.testing_utils import layer_multi_io_test


@test_combinations.run_all_keras_modes
class TestNearestInterpolation(test_combinations.TestCase):
    def setUp(self):
        super(TestNearestInterpolation, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestNearestInterpolation, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        layer_multi_io_test(
            NearestInterpolation,
            kwargs={'scale': None},
            input_shapes=[(2, 16, 16, 10), (2, 24, 32, 3)],
            input_dtypes=['float32', 'float32'],
            expected_output_shapes=[(None, 24, 32, 10)],
            expected_output_dtypes=['float32']
        )
        test_utils.layer_test(
            NearestInterpolation,
            kwargs={'scale': 2},
            input_shape=(2, 16, 16, 10),
            input_dtype='float32',
            expected_output_shape=(None, 32, 32, 10),
            expected_output_dtype='float32'
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        layer_multi_io_test(
            NearestInterpolation,
            kwargs={'scale': None},
            input_shapes=[(2, 16, 16, 10), (2, 24, 32, 3)],
            input_dtypes=['float16', 'float16'],
            expected_output_shapes=[(None, 24, 32, 10)],
            expected_output_dtypes=['float16']
        )
        test_utils.layer_test(
            NearestInterpolation,
            kwargs={'scale': 0.5},
            input_shape=(2, 16, 16, 10),
            input_dtype='float16',
            expected_output_shape=(None, 8, 8, 10),
            expected_output_dtype='float16'
        )


if __name__ == '__main__':
    tf.test.main()
