import tensorflow as tf
from tf_keras import mixed_precision
from tf_keras.src.testing_infra import test_combinations, test_utils
from segme.common.mapool import MultiHeadAttentionPooling


@test_combinations.run_all_keras_modes
class TestMultiheadAttentionPooling(test_combinations.TestCase):
    def setUp(self):
        super(TestMultiheadAttentionPooling, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestMultiheadAttentionPooling, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            MultiHeadAttentionPooling,
            kwargs={'heads': 8, 'queries': 1},
            input_shape=[2, 50, 768],
            input_dtype='float32',
            expected_output_shape=[None, 1, 768],
            expected_output_dtype='float32'
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        test_utils.layer_test(
            MultiHeadAttentionPooling,
            kwargs={'heads': 8, 'queries': 2},
            input_shape=[2, 7, 7, 768],
            input_dtype='float16',
            expected_output_shape=[None, 2, 768],
            expected_output_dtype='float16'
        )


if __name__ == '__main__':
    tf.test.main()
