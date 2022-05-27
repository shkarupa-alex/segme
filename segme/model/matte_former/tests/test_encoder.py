import tensorflow as tf
from keras import layers
from keras.testing_infra import test_combinations
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from ..encoder import Encoder
from ....testing_utils import layer_multi_io_test


@test_combinations.run_all_keras_modes
class TestEncoder(test_combinations.TestCase):
    def test_layer(self):
        layer_multi_io_test(
            Encoder,
            kwargs={},
            input_shapes=[(2, 512, 512, 3), (2, 512, 512, 1)],
            input_dtypes=['uint8'] * 2,
            expected_output_shapes=[
                (None, 512, 512, 32), (None, 256, 256, 32), (None, 128, 128, 64),
                (None, 64, 64, 128), (None, 32, 32, 256), (None, 16, 16, 512)],
            expected_output_dtypes=['float32'] * 6
        )


if __name__ == '__main__':
    tf.test.main()
