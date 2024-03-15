import tensorflow as tf
from tf_keras import mixed_precision
from tf_keras.src.testing_infra import test_combinations
from segme.model.matting.fba_matting.encoder import Encoder
from segme.testing_utils import layer_multi_io_test


@test_combinations.run_all_keras_modes
class TestEncoder(test_combinations.TestCase):
    def setUp(self):
        super(TestEncoder, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestEncoder, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        layer_multi_io_test(
            Encoder,
            kwargs={},
            input_shapes=[(2, 512, 512, 11)],
            input_dtypes=['uint8'],
            expected_output_shapes=[(None, 256, 256, 64), (None, 128, 128, 256), (None, 64, 64, 2048)],
            expected_output_dtypes=['float32'] * 3
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        layer_multi_io_test(
            Encoder,
            kwargs={},
            input_shapes=[(2, 512, 512, 11)],
            input_dtypes=['uint8'],
            expected_output_shapes=[(None, 256, 256, 64), (None, 128, 128, 256), (None, 64, 64, 2048)],
            expected_output_dtypes=['float16'] * 3
        )


if __name__ == '__main__':
    tf.test.main()
