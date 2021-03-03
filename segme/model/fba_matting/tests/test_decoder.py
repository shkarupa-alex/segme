import tensorflow as tf
from tensorflow.python.keras import keras_parameterized
from ..decoder import Decoder
from ....testing_utils import layer_multi_io_test


@keras_parameterized.run_all_keras_modes
class TestDecoder(keras_parameterized.TestCase):
    def setUp(self):
        super(TestDecoder, self).setUp()
        self.default_policy = tf.keras.mixed_precision.experimental.global_policy()

    def tearDown(self):
        super(TestDecoder, self).tearDown()
        tf.keras.mixed_precision.experimental.set_policy(self.default_policy)

    def test_layer(self):
        layer_multi_io_test(
            Decoder,
            kwargs={'pool_scales': (1, 2, 3, 6)},
            input_shapes=[(2, 128, 128, 4), (2, 64, 64, 6), (2, 32, 32, 8), (2, 256, 256, 3), (2, 256, 256, 2)],
            input_dtypes=['float32'] * 5,
            expected_output_shapes=[(None, 256, 256, 7)],
            expected_output_dtypes=['float32']
        )

        tf.keras.mixed_precision.experimental.set_policy('mixed_float16')
        layer_multi_io_test(
            Decoder,
            kwargs={'pool_scales': (1, 2, 3, 6)},
            input_shapes=[(2, 128, 128, 4), (2, 64, 64, 6), (2, 32, 32, 8), (2, 256, 256, 3), (2, 256, 256, 2)],
            input_dtypes=['float16'] * 5,
            expected_output_shapes=[(None, 256, 256, 7)],
            expected_output_dtypes=['float32']
        )
        tf.keras.mixed_precision.experimental.set_policy(self.default_policy)

if __name__ == '__main__':
    tf.test.main()
