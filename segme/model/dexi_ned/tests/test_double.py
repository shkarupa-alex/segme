import tensorflow as tf
from keras import keras_parameterized, testing_utils
from ..double import DoubleConvBlock


@keras_parameterized.run_all_keras_modes
class TestDoubleConvBlock(keras_parameterized.TestCase):
    def test_layer(self):
        testing_utils.layer_test(
            DoubleConvBlock,
            kwargs={'mid_features': 10},
            input_shape=[2, 16, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 10],
            expected_output_dtype='float32'
        )
        testing_utils.layer_test(
            DoubleConvBlock,
            kwargs={'mid_features': 10, 'out_features': 5, 'stride': 2},
            input_shape=[2, 16, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 8, 8, 5],
            expected_output_dtype='float32'
        )


if __name__ == '__main__':
    tf.test.main()
