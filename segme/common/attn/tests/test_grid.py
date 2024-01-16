import tensorflow as tf
from keras import mixed_precision
from keras.src.testing_infra import test_combinations, test_utils
from segme.common.attn.grid import GridAttention


@test_combinations.run_all_keras_modes
class TestGridAttention(test_combinations.TestCase):
    def setUp(self):
        super(TestGridAttention, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestGridAttention, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            GridAttention,
            kwargs={
                'current_window': 4, 'pretrain_window': 4, 'num_heads': 2, 'qk_units': None, 'qkv_bias': True,
                'cpb_units': 512, 'proj_bias': True},
            input_shape=[2, 15, 17, 4],
            input_dtype='float32',
            expected_output_shape=[None, 15, 17, 4],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            GridAttention,
            kwargs={
                'current_window': 8, 'pretrain_window': 4, 'num_heads': 2, 'qk_units': None, 'qkv_bias': True,
                'cpb_units': 512, 'proj_bias': True},
            input_shape=[2, 14, 18, 4],
            input_dtype='float32',
            expected_output_shape=[None, 14, 18, 4],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            GridAttention,
            kwargs={
                'current_window': 4, 'pretrain_window': 4, 'num_heads': 4, 'qk_units': None, 'qkv_bias': True,
                'cpb_units': 512, 'proj_bias': True},
            input_shape=[2, 16, 16, 4],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 4],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            GridAttention,
            kwargs={
                'current_window': 4, 'pretrain_window': 4, 'num_heads': 2, 'qk_units': 4, 'qkv_bias': True,
                'cpb_units': 512, 'proj_bias': True},
            input_shape=[2, 15, 17, 4],
            input_dtype='float32',
            expected_output_shape=[None, 15, 17, 4],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            GridAttention,
            kwargs={
                'current_window': 4, 'pretrain_window': 4, 'num_heads': 2, 'qk_units': None, 'qkv_bias': False,
                'cpb_units': 512, 'proj_bias': True},
            input_shape=[2, 15, 17, 4],
            input_dtype='float32',
            expected_output_shape=[None, 15, 17, 4],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            GridAttention,
            kwargs={
                'current_window': 4, 'pretrain_window': 4, 'num_heads': 2, 'qk_units': None, 'qkv_bias': True,
                'cpb_units': 384, 'proj_bias': True},
            input_shape=[2, 15, 17, 4],
            input_dtype='float32',
            expected_output_shape=[None, 15, 17, 4],
            expected_output_dtype='float32'
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        test_utils.layer_test(
            GridAttention,
            kwargs={
                'current_window': 6, 'pretrain_window': 4, 'num_heads': 2, 'qk_units': None, 'qkv_bias': True,
                'cpb_units': 512, 'proj_bias': False},
            input_shape=[2, 16, 16, 4],
            input_dtype='float16',
            expected_output_shape=[None, 16, 16, 4],
            expected_output_dtype='float16'
        )


if __name__ == '__main__':
    tf.test.main()
