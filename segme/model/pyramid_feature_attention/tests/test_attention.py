import tensorflow as tf
from tensorflow.python.keras import keras_parameterized, testing_utils
from ..attention import SpatialAttention, ChannelWiseAttention


@keras_parameterized.run_all_keras_modes
class TestSpatialAttention(keras_parameterized.TestCase):
    def test_layer(self):
        testing_utils.layer_test(
            SpatialAttention,
            kwargs={},
            input_shape=[2, 16, 16, 8],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 8],
            expected_output_dtype='float32'
        )


@keras_parameterized.run_all_keras_modes
class TestChannelWiseAttention(keras_parameterized.TestCase):
    def test_layer(self):
        testing_utils.layer_test(
            ChannelWiseAttention,
            kwargs={},
            input_shape=[2, 16, 16, 8],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 8],
            expected_output_dtype='float32'
        )


if __name__ == '__main__':
    tf.test.main()
