import tensorflow as tf
from tensorflow.python.keras import keras_parameterized, testing_utils
from ..single import SingleConvBlock


@keras_parameterized.run_all_keras_modes
class TestSingleConvBlock(keras_parameterized.TestCase):
    def test_layer(self):
        testing_utils.layer_test(
            SingleConvBlock,
            kwargs={'out_features': 10},
            input_shape=[2, 16, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 10],
            expected_output_dtype='float32'
        )

        const_init = tf.keras.initializers.constant(0.1)
        testing_utils.layer_test(
            SingleConvBlock,
            kwargs={'out_features': 5, 'kernel_size': 2, 'stride': 2, 'weight_norm': True,
                    'kernel_initializer': const_init},
            input_shape=[2, 16, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 8, 8, 5],
            expected_output_dtype='float32'
        )


if __name__ == '__main__':
    tf.test.main()
