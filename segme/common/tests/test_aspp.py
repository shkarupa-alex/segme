import tensorflow as tf
from tensorflow.python.keras import keras_parameterized, testing_utils
from ..aspp import ASPPPool2D, ASPP2D


@keras_parameterized.run_all_keras_modes
class TestASPPPool2D(keras_parameterized.TestCase):
    def test_layer(self):
        testing_utils.layer_test(
            ASPPPool2D,
            kwargs={'filters': 10},
            input_shape=[2, 16, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 10],
            expected_output_dtype='float32'
        )


@keras_parameterized.run_all_keras_modes
class TestASPP2D(keras_parameterized.TestCase):
    def test_layer(self):
        testing_utils.layer_test(
            ASPP2D,
            kwargs={'filters': 10, 'stride': 8},
            input_shape=[2, 16, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 10],
            expected_output_dtype='float32'
        )
        testing_utils.layer_test(
            ASPP2D,
            kwargs={'filters': 10, 'stride': 16},
            input_shape=[2, 16, 16, 5],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 10],
            expected_output_dtype='float32'
        )
        testing_utils.layer_test(
            ASPP2D,
            kwargs={'filters': 10, 'stride': 32},
            input_shape=[2, 16, 16, 1],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 10],
            expected_output_dtype='float32'
        )


if __name__ == '__main__':
    tf.test.main()
