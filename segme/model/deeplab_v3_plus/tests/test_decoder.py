import tensorflow as tf
from tensorflow.python.keras import keras_parameterized
from ..decoder import Decoder
from ....testing_utils import layer_multi_io_test


@keras_parameterized.run_all_keras_modes
class TestDecoder(keras_parameterized.TestCase):
    def test_layer(self):
        layer_multi_io_test(
            Decoder,
            kwargs={'low_filters': 8, 'decoder_filters': 4},
            input_shapes=[(2, 64, 64, 10), (2, 16, 16, 3), (2, 4, 4, 3)],
            input_dtypes=['uint8', 'float32', 'float32'],
            expected_output_shapes=[(None, 64, 64, 4)],
            expected_output_dtypes=['float32']

        )


if __name__ == '__main__':
    tf.test.main()
