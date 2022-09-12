import tensorflow as tf
from keras.mixed_precision import policy as mixed_precision
from keras.testing_infra import test_combinations, test_utils
from segme.common.pad import SymmetricPadding


@test_combinations.run_all_keras_modes
class TestSymmetricPadding(test_combinations.TestCase):
    def setUp(self):
        super(TestSymmetricPadding, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestSymmetricPadding, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            SymmetricPadding,
            kwargs={'padding': 1},
            input_shape=[2, 4, 5, 3],
            input_dtype='float32',
            expected_output_shape=[None, 6, 7, 3],
            expected_output_dtype='float32'
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        test_utils.layer_test(
            SymmetricPadding,
            kwargs={'padding': 1},
            input_shape=[2, 4, 5, 3],
            input_dtype='float16',
            expected_output_shape=[None, 6, 7, 3],
            expected_output_dtype='float16'
        )

    def test_error(self):
        with self.assertRaisesRegex(ValueError, 'Symmetric padding can lead to misbehavior'):
            SymmetricPadding(((0, 1), (1, 2)))


if __name__ == '__main__':
    tf.test.main()
