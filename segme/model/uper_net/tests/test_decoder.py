import tensorflow as tf
from keras.testing_infra import test_combinations
from segme.model.uper_net.decoder import Decoder
from segme.testing_utils import layer_multi_io_test


@test_combinations.run_all_keras_modes
class TestDecoder(test_combinations.TestCase):
    def test_layer(self):
        layer_multi_io_test(
            Decoder,
            kwargs={'filters': 6},
            input_shapes=[(2, 96, 96, 2), (2, 48, 48, 3), (2, 24, 24, 4), (2, 12, 12, 5)],
            input_dtypes=['float32'] * 4,
            expected_output_shapes=[(None, 96, 96, 6)],
            expected_output_dtypes=['float32']
        )


if __name__ == '__main__':
    tf.test.main()
