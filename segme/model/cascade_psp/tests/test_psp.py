import tensorflow as tf
from tensorflow.python.keras import keras_parameterized, testing_utils
from ..psp import PSP


@keras_parameterized.run_all_keras_modes
class TestPSP(keras_parameterized.TestCase):
    def test_layer(self):
        testing_utils.layer_test(
            PSP,
            kwargs={'filters': 2, 'sizes': (1, 2, 3, 6)},
            input_shape=[2, 18, 18, 3],
            input_dtype='float32',
            expected_output_shape=[None, 18, 18, 2],
            expected_output_dtype='float32'
        )


if __name__ == '__main__':
    tf.test.main()
