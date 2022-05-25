import tensorflow as tf
from keras.testing_infra import test_combinations
from ..decoder import Decoder
from ....testing_utils import layer_multi_io_test


@test_combinations.run_all_keras_modes
class TestDecoder(test_combinations.TestCase):
    def test_layer(self):
        layer_multi_io_test(
            Decoder,
            kwargs={'refine': False, 'filters': 7},
            input_shapes=[(2, 32, 32, 3), (2, 16, 16, 4), (2, 8, 8, 5), (2, 4, 4, 6)],
            input_dtypes=['float32'] * 4,
            expected_output_shapes=[
                (None, 32, 32, 7), (None, 16, 16, 7), (None, 8, 8, 7), (None, 4, 4, 6), (None, 32, 32, 7)],
            expected_output_dtypes=['float32'] * 5

        )


if __name__ == '__main__':
    tf.test.main()
