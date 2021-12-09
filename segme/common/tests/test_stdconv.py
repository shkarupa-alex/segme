import numpy as np
import tensorflow as tf
from keras import keras_parameterized, testing_utils
from keras.mixed_precision import policy as mixed_precision
from ..stdconv import StandardizedConv2D


@keras_parameterized.run_all_keras_modes
class TestStandardizedConv2D(keras_parameterized.TestCase):
    def setUp(self):
        super(TestStandardizedConv2D, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestStandardizedConv2D, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        # testing_utils.layer_test(
        #     StandardizedConv2D,
        #     kwargs={'filters': 4, 'kernel_size': 1, 'strides': 1, 'padding': 'valid'},
        #     input_shape=[2, 16, 16, 8],
        #     input_dtype='float32',
        #     expected_output_shape=[None, 16, 16, 4],
        #     expected_output_dtype='float32'
        # )

        mixed_precision.set_global_policy('mixed_float16')
        result = testing_utils.layer_test(
            StandardizedConv2D,
            kwargs={'filters': 4, 'kernel_size': 3, 'strides': 2, 'padding': 'same'},
            input_shape=[2, 16, 16, 8],
            input_dtype='float16',
            expected_output_shape=[None, 8, 8, 4],
            expected_output_dtype='float16'
        )
        self.assertTrue(np.all(np.isfinite(result)))
        self.assertFalse(True)


if __name__ == '__main__':
    tf.test.main()
