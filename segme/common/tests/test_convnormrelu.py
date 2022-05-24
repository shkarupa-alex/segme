import tensorflow as tf
from keras.testing_infra import test_combinations, test_utils
from ..convnormrelu import ConvNormRelu, DepthwiseConvNormRelu


@test_combinations.run_all_keras_modes
class TestConvNormRelu(test_combinations.TestCase):
    def test_layer(self):
        test_utils.layer_test(
            ConvNormRelu,
            kwargs={'filters': 4, 'kernel_size': 3, 'strides': 2},
            input_shape=[2, 16, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 8, 8, 4],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            ConvNormRelu,
            kwargs={'filters': 64, 'kernel_size': 2, 'dilation_rate': 2, 'standardized': True},
            input_shape=[2, 17, 17, 3],
            input_dtype='float32',
            expected_output_shape=[None, 17, 17, 64],
            expected_output_dtype='float32'
        )


@test_combinations.run_all_keras_modes
class TestDepthwiseConvNormRelu(test_combinations.TestCase):
    def test_layer(self):
        test_utils.layer_test(
            DepthwiseConvNormRelu,
            kwargs={'kernel_size': 3, 'strides': 2},
            input_shape=[2, 16, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 8, 8, 3],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            DepthwiseConvNormRelu,
            kwargs={'kernel_size': 2, 'dilation_rate': 2, 'standardized': True},
            input_shape=[2, 17, 17, 64],
            input_dtype='float32',
            expected_output_shape=[None, 17, 17, 64],
            expected_output_dtype='float32'
        )


if __name__ == '__main__':
    tf.test.main()
