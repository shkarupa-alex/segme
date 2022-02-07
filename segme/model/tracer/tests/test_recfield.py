import tensorflow as tf
from keras import keras_parameterized, testing_utils
from ..recfield import ReceptiveField


@keras_parameterized.run_all_keras_modes
class TestReceptiveField(keras_parameterized.TestCase):
    def test_layer(self):
        testing_utils.layer_test(
            ReceptiveField,
            kwargs={'filters': 7},
            input_shape=(2, 32, 32, 16),
            input_dtype='float32',
            expected_output_shape=(None, 32, 32, 7),
            expected_output_dtype='float32'
        )


if __name__ == '__main__':
    tf.test.main()
