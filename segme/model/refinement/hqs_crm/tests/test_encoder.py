import tensorflow as tf
from keras.testing_infra import test_combinations
from segme.model.refinement.hqs_crm.encoder import Encoder
from segme.testing_utils import layer_multi_io_test


@test_combinations.run_all_keras_modes
class TestEncoder(test_combinations.TestCase):
    def test_layer(self):
        layer_multi_io_test(
            Encoder,
            kwargs={},
            input_shapes=[(2, 256, 256, 4)],
            input_dtypes=['uint8'],
            expected_output_shapes=[(None, 128, 128, 64), (None, 64, 64, 256), (None, 32, 32, 2048)],
            expected_output_dtypes=['float32'] * 3
        )


if __name__ == '__main__':
    tf.test.main()
