import tensorflow as tf
from keras.testing_infra import test_combinations
from ..decoder import Decoder
from ....testing_utils import layer_multi_io_test


@test_combinations.run_all_keras_modes
class TestDecoder(test_combinations.TestCase):
    def test_layer(self):
        layer_multi_io_test(
            Decoder,
            kwargs={'low_filters': 8, 'decoder_filters': 4},
            input_shapes=[(2, 16, 16, 3), (2, 4, 4, 3)],
            input_dtypes=['float32', 'float32'],
            expected_output_shapes=[(None, 16, 16, 4)],
            expected_output_dtypes=['float32']

        )


if __name__ == '__main__':
    tf.test.main()
