import tensorflow as tf
from tensorflow.python.keras import keras_parameterized, testing_utils
from ..convbnrelu import ConvBnRelu


@keras_parameterized.run_all_keras_modes
class TestConvBnRelu(keras_parameterized.TestCase):
    def test_layer(self):
        testing_utils.layer_test(
            ConvBnRelu,
            kwargs={'filters': 4, 'kernel_size': 3, 'strides': 2},
            input_shape=[2, 16, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 8, 8, 4],
            expected_output_dtype='float32'
        )
        testing_utils.layer_test(
            ConvBnRelu,
            kwargs={'filters': 10, 'kernel_size': 1, 'dilation_rate': 2},
            input_shape=[2, 17, 17, 3],
            input_dtype='float32',
            expected_output_shape=[None, 17, 17, 10],
            expected_output_dtype='float32'
        )


if __name__ == '__main__':
    tf.test.main()
