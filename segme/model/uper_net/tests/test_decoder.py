import tensorflow as tf
from keras import keras_parameterized
from ..decoder import Decoder
from ....testing_utils import layer_multi_io_test


@keras_parameterized.run_all_keras_modes
class TestDecoder(keras_parameterized.TestCase):
    def test_layer(self):
        layer_multi_io_test(
            Decoder,
            kwargs={'filters': 6, 'psp_sizes': (1, 2, 3, 6)},
            input_shapes=[(2, 96, 96, 2), (2, 48, 48, 3), (2, 24, 24, 4), (2, 12, 12, 5)],
            input_dtypes=['float32'] * 4,
            expected_output_shapes=[(None, 96, 96, 6)],
            expected_output_dtypes=['float32']
        )


if __name__ == '__main__':
    tf.test.main()
