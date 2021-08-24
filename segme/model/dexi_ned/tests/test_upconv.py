import tensorflow as tf
from keras import keras_parameterized, testing_utils
from ..upconv import UpConvBlock


@keras_parameterized.run_all_keras_modes
class TestUpConvBlock(keras_parameterized.TestCase):
    def test_layer(self):
        testing_utils.layer_test(
            UpConvBlock,
            kwargs={'filters': 1, 'up_scale': 2},
            input_shape=[2, 16, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 64, 64, 1],
            expected_output_dtype='float32'
        )
        testing_utils.layer_test(
            UpConvBlock,
            kwargs={'filters': 2, 'up_scale': 2},
            input_shape=[2, 16, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 64, 64, 2],
            expected_output_dtype='float32'
        )


if __name__ == '__main__':
    tf.test.main()
