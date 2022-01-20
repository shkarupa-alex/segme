import tensorflow as tf
from keras import keras_parameterized, testing_utils
from ..aspp import ASPPPool, ASPP


@keras_parameterized.run_all_keras_modes
class TestASPPPool(keras_parameterized.TestCase):
    def test_layer(self):
        testing_utils.layer_test(
            ASPPPool,
            kwargs={'filters': 10},
            input_shape=[2, 16, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 10],
            expected_output_dtype='float32'
        )


@keras_parameterized.run_all_keras_modes
class TestASPP(keras_parameterized.TestCase):
    def test_layer(self):
        testing_utils.layer_test(
            ASPP,
            kwargs={'filters': 10, 'stride': 8},
            input_shape=[2, 16, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 10],
            expected_output_dtype='float32'
        )
        testing_utils.layer_test(
            ASPP,
            kwargs={'filters': 10, 'stride': 16},
            input_shape=[2, 16, 16, 5],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 10],
            expected_output_dtype='float32'
        )
        testing_utils.layer_test(
            ASPP,
            kwargs={'filters': 10, 'stride': 32},
            input_shape=[2, 16, 16, 1],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 10],
            expected_output_dtype='float32'
        )


if __name__ == '__main__':
    tf.test.main()
