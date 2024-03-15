import tensorflow as tf
from tf_keras import mixed_precision
from tf_keras.src.testing_infra import test_combinations, test_utils
from segme.common.attn.slide import SlideAttention


@test_combinations.run_all_keras_modes
class TestSlideAttention(test_combinations.TestCase):
    def setUp(self):
        super(TestSlideAttention, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestSlideAttention, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            SlideAttention,
            kwargs={
                'window_size': 3, 'num_heads': 2, 'qk_units': None, 'qkv_bias': True, 'cpb_units': 512,
                'dilation_rate': 1, 'proj_bias': True},
            input_shape=[2, 15, 17, 4],
            input_dtype='float32',
            expected_output_shape=[None, 15, 17, 4],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            SlideAttention,
            kwargs={
                'window_size': 5, 'num_heads': 2, 'qk_units': None, 'qkv_bias': True, 'cpb_units': 512,
                'dilation_rate': 1, 'proj_bias': True},
            input_shape=[2, 15, 17, 4],
            input_dtype='float32',
            expected_output_shape=[None, 15, 17, 4],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            SlideAttention,
            kwargs={
                'window_size': 3, 'num_heads': 4, 'qk_units': None, 'qkv_bias': True, 'cpb_units': 512,
                'dilation_rate': 1, 'proj_bias': True},
            input_shape=[2, 15, 17, 4],
            input_dtype='float32',
            expected_output_shape=[None, 15, 17, 4],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            SlideAttention,
            kwargs={
                'window_size': 3, 'num_heads': 2, 'qk_units': 1, 'qkv_bias': True, 'cpb_units': 512, 'dilation_rate': 1,
                'proj_bias': True},
            input_shape=[2, 15, 17, 4],
            input_dtype='float32',
            expected_output_shape=[None, 15, 17, 4],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            SlideAttention,
            kwargs={
                'window_size': 3, 'num_heads': 2, 'qk_units': None, 'qkv_bias': False, 'cpb_units': 512,
                'dilation_rate': 1, 'proj_bias': True},
            input_shape=[2, 15, 17, 4],
            input_dtype='float32',
            expected_output_shape=[None, 15, 17, 4],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            SlideAttention,
            kwargs={
                'window_size': 3, 'num_heads': 2, 'qk_units': None, 'qkv_bias': True, 'cpb_units': 384,
                'dilation_rate': 1, 'proj_bias': True},
            input_shape=[2, 15, 17, 4],
            input_dtype='float32',
            expected_output_shape=[None, 15, 17, 4],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            SlideAttention,
            kwargs={
                'window_size': 3, 'num_heads': 2, 'qk_units': None, 'qkv_bias': True, 'cpb_units': 512,
                'dilation_rate': 2, 'proj_bias': True},
            input_shape=[2, 15, 17, 4],
            input_dtype='float32',
            expected_output_shape=[None, 15, 17, 4],
            expected_output_dtype='float32'
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        test_utils.layer_test(
            SlideAttention,
            kwargs={
                'window_size': 3, 'num_heads': 2, 'qk_units': None, 'qkv_bias': True, 'cpb_units': 512,
                'dilation_rate': 1, 'proj_bias': False},
            input_shape=[2, 15, 17, 4],
            input_dtype='float16',
            expected_output_shape=[None, 15, 17, 4],
            expected_output_dtype='float16'
        )


if __name__ == '__main__':
    tf.test.main()
