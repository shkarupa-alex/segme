import tensorflow as tf
from keras import keras_parameterized, testing_utils
from ..enhance import CAEnhance, SAEnhance, CASAEnhance
from ....testing_utils import layer_multi_io_test


@keras_parameterized.run_all_keras_modes
class TestCAEnhance(keras_parameterized.TestCase):
    def test_layer(self):
        layer_multi_io_test(
            CAEnhance,
            kwargs={},
            input_shapes=[(2, 32, 32, 18), (2, 32, 32, 18)],
            input_dtypes=['float32'] * 2,
            expected_output_shapes=[(None, 32, 32, 18)],
            expected_output_dtypes=['float32']
        )


@keras_parameterized.run_all_keras_modes
class TestSAEnhance(keras_parameterized.TestCase):
    def test_layer(self):
        testing_utils.layer_test(
            SAEnhance,
            kwargs={},
            input_shape=[2, 16, 16, 4],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 1],
            expected_output_dtype='float32'
        )


@keras_parameterized.run_all_keras_modes
class TestCASAEnhance(keras_parameterized.TestCase):
    def test_layer(self):
        layer_multi_io_test(
            CASAEnhance,
            kwargs={},
            input_shapes=[(2, 32, 32, 18), (2, 32, 32, 18)],
            input_dtypes=['float32'] * 2,
            expected_output_shapes=[(None, 32, 32, 18)],
            expected_output_dtypes=['float32']
        )


if __name__ == '__main__':
    tf.test.main()
