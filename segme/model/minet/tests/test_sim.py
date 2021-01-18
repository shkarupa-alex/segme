import tensorflow as tf
from tensorflow.python.keras import keras_parameterized, testing_utils
from ..sim import SIM


@keras_parameterized.run_all_keras_modes
class TestSim(keras_parameterized.TestCase):
    def test_layer(self):
        testing_utils.layer_test(
            SIM,
            kwargs={'filters': 4},
            input_shape=[2, 16, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 3],
            expected_output_dtype='float32'
        )
        testing_utils.layer_test(
            SIM,
            kwargs={'filters': 10},
            input_shape=[2, 17, 17, 3],
            input_dtype='float32',
            expected_output_shape=[None, 17, 17, 3],
            expected_output_dtype='float32'
        )


if __name__ == '__main__':
    tf.test.main()
