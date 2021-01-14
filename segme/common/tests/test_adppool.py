import tensorflow as tf
from tensorflow.python.keras import keras_parameterized, testing_utils
from ..adppool import AdaptiveAveragePooling, AdaptiveMaxPooling


@keras_parameterized.run_all_keras_modes
class TestAdaptiveAveragePooling(keras_parameterized.TestCase):
    def test_layer(self):
        testing_utils.layer_test(
            AdaptiveAveragePooling,
            kwargs={'output_size': 2},
            input_shape=[2, 16, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 2, 2, 3],
            expected_output_dtype='float32'
        )
        testing_utils.layer_test(
            AdaptiveAveragePooling,
            kwargs={'output_size': (4, 3)},
            input_shape=[2, 15, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 4, 3, 3],
            expected_output_dtype='float32'
        )


@keras_parameterized.run_all_keras_modes
class TestAdaptiveMaxPooling(keras_parameterized.TestCase):
    def test_layer(self):
        testing_utils.layer_test(
            AdaptiveMaxPooling,
            kwargs={'output_size': 2},
            input_shape=[2, 16, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 2, 2, 3],
            expected_output_dtype='float32'
        )
        testing_utils.layer_test(
            AdaptiveMaxPooling,
            kwargs={'output_size': (4, 3)},
            input_shape=[2, 15, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 4, 3, 3],
            expected_output_dtype='float32'
        )


if __name__ == '__main__':
    tf.test.main()
