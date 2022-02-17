import tensorflow as tf
from keras import keras_parameterized
from ..decoder import Decoder
from ....testing_utils import layer_multi_io_test


@keras_parameterized.run_all_keras_modes
class TestDecoder(keras_parameterized.TestCase):
    def test_layer(self):
        layer_multi_io_test(
            Decoder,
            kwargs={'standardized': False},
            input_shapes=[(2, 32, 32, 64), (2, 16, 16, 96), (2, 8, 8, 128), (2, 4, 4, 160)],
            input_dtypes=['float32'] * 4,
            expected_output_shapes=[(None, 32, 32, 64)],
            expected_output_dtypes=['float32']
        )
        layer_multi_io_test(
            Decoder,
            kwargs={'standardized': True},
            input_shapes=[(2, 32, 32, 64), (2, 16, 16, 96), (2, 8, 8, 128), (2, 4, 4, 160)],
            input_dtypes=['float32'] * 4,
            expected_output_shapes=[(None, 32, 32, 64)],
            expected_output_dtypes=['float32']
        )


if __name__ == '__main__':
    tf.test.main()
