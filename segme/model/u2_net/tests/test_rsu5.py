import tensorflow as tf
from tensorflow.python.keras import keras_parameterized, testing_utils
from ..rsu5 import RSU5


@keras_parameterized.run_all_keras_modes
class TestRSU7(keras_parameterized.TestCase):
    def test_layer(self):
        testing_utils.layer_test(
            RSU5,
            kwargs={'mid_features': 5, 'out_features': 4},
            input_shape=[2, 16, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 4],
            expected_output_dtype='float32'
        )


if __name__ == '__main__':
    tf.test.main()
