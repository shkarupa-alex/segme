import tensorflow as tf
from keras import keras_parameterized, testing_utils
from ..convnormrelu import ConvNormRelu


@keras_parameterized.run_all_keras_modes
class TestConvNormRelu(keras_parameterized.TestCase):
    def test_layer(self):
        testing_utils.layer_test(
            ConvNormRelu,
            kwargs={'filters': 4, 'kernel_size': 3, 'strides': 2},
            input_shape=[2, 16, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 7, 7, 4],
            expected_output_dtype='float32'
        )
        testing_utils.layer_test(
            ConvNormRelu,
            kwargs={'filters': 64, 'kernel_size': 2, 'padding': 'same', 'dilation_rate': 2, 'standardized': True},
            input_shape=[2, 17, 17, 3],
            input_dtype='float32',
            expected_output_shape=[None, 17, 17, 64],
            expected_output_dtype='float32'
        )


if __name__ == '__main__':
    tf.test.main()
