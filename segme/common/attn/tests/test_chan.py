import tensorflow as tf
from keras import mixed_precision
from keras.src.testing_infra import test_combinations, test_utils
from segme.common.attn.chan import ChannelAttention


@test_combinations.run_all_keras_modes
class TestChannelAttention(test_combinations.TestCase):
    def setUp(self):
        super(TestChannelAttention, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestChannelAttention, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            ChannelAttention,
            kwargs={'num_heads': 2, 'qkv_bias': True, 'proj_bias': True},
            input_shape=[2, 16, 16, 4],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 4],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            ChannelAttention,
            kwargs={'num_heads': 2, 'qkv_bias': False, 'proj_bias': True},
            input_shape=[2, 16, 16, 4],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 4],
            expected_output_dtype='float32'
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        test_utils.layer_test(
            ChannelAttention,
            kwargs={'num_heads': 4, 'qkv_bias': True, 'proj_bias': False},
            input_shape=[2, 16, 16, 4],
            input_dtype='float16',
            expected_output_shape=[None, 16, 16, 4],
            expected_output_dtype='float16'
        )


if __name__ == '__main__':
    tf.test.main()
