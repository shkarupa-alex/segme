import tensorflow as tf
from keras.testing_infra import test_combinations
from keras.mixed_precision import policy as mixed_precision
from ..decoder import Decoder
from ....testing_utils import layer_multi_io_test


@test_combinations.run_all_keras_modes
class TestDecoder(test_combinations.TestCase):
    def setUp(self):
        super(TestDecoder, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestDecoder, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        layer_multi_io_test(
            Decoder,
            kwargs={'pool_scales': (1, 2, 3, 6)},
            input_shapes=[
                (2, 128, 128, 4), (2, 64, 64, 6), (2, 32, 32, 8), (2, 256, 256, 3), (2, 256, 256, 3), (2, 256, 256, 2)],
            input_dtypes=['float32'] * 6,
            expected_output_shapes=[(None, 256, 256, 7)],
            expected_output_dtypes=['float32']
        )

        mixed_precision.set_global_policy('mixed_float16')
        layer_multi_io_test(
            Decoder,
            kwargs={'pool_scales': (1, 2, 3, 6)},
            input_shapes=[
                (2, 128, 128, 4), (2, 64, 64, 6), (2, 32, 32, 8), (2, 256, 256, 3), (2, 256, 256, 3), (2, 256, 256, 2)],
            input_dtypes=['float16'] * 6,
            expected_output_shapes=[(None, 256, 256, 7)],
            expected_output_dtypes=['float32']
        )


if __name__ == '__main__':
    tf.test.main()
