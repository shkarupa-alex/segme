import tensorflow as tf
from keras.testing_infra import test_combinations
from segme.model.refinement.hqs_crm.decoder import Decoder
from segme.testing_utils import layer_multi_io_test


@test_combinations.run_all_keras_modes
class TestDecoder(test_combinations.TestCase):
    def test_layer(self):
        layer_multi_io_test(
            Decoder,
            kwargs={'aspp_filters': (64, 64, 128), 'aspp_drop': 0.5, 'mlp_units': (32, 32, 32, 32)},
            input_shapes=[(3, 128, 128, 64), (3, 64, 64, 256), (3, 32, 32, 2048), (3, 96, 96, 2)],
            input_dtypes=['float32'] * 4,
            expected_output_shapes=[(None, 96, 96, 1)],
            expected_output_dtypes=['float32']
        )


if __name__ == '__main__':
    tf.test.main()
