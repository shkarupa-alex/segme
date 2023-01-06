import tensorflow as tf
from keras import layers, models
from keras.mixed_precision import policy as mixed_precision
from keras.testing_infra import test_combinations, test_utils
from segme.policy.backbone.diy.coma.attn import DHMSA, CHMSA, GGMSA


@test_combinations.run_all_keras_modes
class TestDHMSA(test_combinations.TestCase):
    def setUp(self):
        super(TestDHMSA, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestDHMSA, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            DHMSA,
            kwargs={'current_window': 4, 'pretrain_window': 4, 'num_heads': 2, 'dilation_rate': 1, 'qkv_bias': True,
                    'attn_drop': 0., 'proj_drop': 0.},
            input_shape=[2, 15, 17, 4],
            input_dtype='float32',
            expected_output_shape=[None, 15, 17, 4],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            DHMSA,
            kwargs={'current_window': 8, 'pretrain_window': 4, 'num_heads': 2, 'dilation_rate': 1, 'qkv_bias': True,
                    'attn_drop': 0., 'proj_drop': 0.},
            input_shape=[2, 14, 18, 4],
            input_dtype='float32',
            expected_output_shape=[None, 14, 18, 4],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            DHMSA,
            kwargs={'current_window': 4, 'pretrain_window': 4, 'num_heads': 4, 'dilation_rate': 1, 'qkv_bias': True,
                    'attn_drop': 0., 'proj_drop': 0.},
            input_shape=[2, 16, 16, 4],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 4],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            DHMSA,
            kwargs={'current_window': 4, 'pretrain_window': 4, 'num_heads': 2, 'dilation_rate': 2, 'qkv_bias': True,
                    'attn_drop': 0., 'proj_drop': 0.},
            input_shape=[2, 13, 19, 4],
            input_dtype='float32',
            expected_output_shape=[None, 13, 19, 4],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            DHMSA,
            kwargs={'current_window': 4, 'pretrain_window': 4, 'num_heads': 2, 'dilation_rate': 1, 'qkv_bias': False,
                    'attn_drop': 0., 'proj_drop': 0.},
            input_shape=[2, 16, 16, 4],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 4],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            DHMSA,
            kwargs={'current_window': 4, 'pretrain_window': 4, 'num_heads': 2, 'dilation_rate': 1, 'qkv_bias': True,
                    'attn_drop': 0.1, 'proj_drop': 0.2},
            input_shape=[2, 16, 16, 4],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 4],
            expected_output_dtype='float32'
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        test_utils.layer_test(
            DHMSA,
            kwargs={'current_window': 6, 'pretrain_window': 4, 'num_heads': 2, 'dilation_rate': 2, 'qkv_bias': True,
                    'attn_drop': 0., 'proj_drop': 0.},
            input_shape=[2, 16, 16, 4],
            input_dtype='float16',
            expected_output_shape=[None, 16, 16, 4],
            expected_output_dtype='float16'
        )

    # def test_jit(self):
    #     inputs = layers.Input(shape=(None, None, 32), dtype='float32')
    #     outputs = DHMSA(8, 8, 4)(inputs)
    #     outputs = layers.Dense(32)(outputs)
    #     model = models.Model(inputs=inputs, outputs=outputs)
    #     model.compile('adam', 'mse', jit_compile=True)
    #
    #     data = tf.random.uniform((16, 12, 12, 32), dtype='float32')
    #     model.fit(data, data)


@test_combinations.run_all_keras_modes
class TestCHMSA(test_combinations.TestCase):
    def setUp(self):
        super(TestCHMSA, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestCHMSA, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            CHMSA,
            kwargs={'num_heads': 2, 'qkv_bias': True, 'attn_drop': 0., 'proj_drop': 0.},
            input_shape=[2, 16, 16, 4],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 4],
            expected_output_dtype='float32'
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        test_utils.layer_test(
            CHMSA,
            kwargs={'num_heads': 4, 'qkv_bias': False, 'attn_drop': 0.1, 'proj_drop': 0.2},
            input_shape=[2, 16, 16, 4],
            input_dtype='float16',
            expected_output_shape=[None, 16, 16, 4],
            expected_output_dtype='float16'
        )


@test_combinations.run_all_keras_modes
class TestGGMSA(test_combinations.TestCase):
    def setUp(self):
        super(TestGGMSA, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestGGMSA, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            GGMSA,
            kwargs={'current_window': 4, 'pretrain_window': 4, 'num_heads': 2, 'qkv_bias': True, 'attn_drop': 0.,
                    'proj_drop': 0.},
            input_shape=[2, 15, 17, 4],
            input_dtype='float32',
            expected_output_shape=[None, 15, 17, 4],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            GGMSA,
            kwargs={'current_window': 8, 'pretrain_window': 4, 'num_heads': 2, 'qkv_bias': True, 'attn_drop': 0.,
                    'proj_drop': 0.},
            input_shape=[2, 14, 18, 4],
            input_dtype='float32',
            expected_output_shape=[None, 14, 18, 4],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            GGMSA,
            kwargs={'current_window': 4, 'pretrain_window': 4, 'num_heads': 4, 'qkv_bias': True, 'attn_drop': 0.,
                    'proj_drop': 0.},
            input_shape=[2, 16, 16, 4],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 4],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            GGMSA,
            kwargs={'current_window': 4, 'pretrain_window': 4, 'num_heads': 2, 'qkv_bias': False, 'attn_drop': 0.,
                    'proj_drop': 0.},
            input_shape=[2, 16, 16, 4],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 4],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            GGMSA,
            kwargs={'current_window': 4, 'pretrain_window': 4, 'num_heads': 2, 'qkv_bias': True, 'attn_drop': 0.1,
                    'proj_drop': 0.2},
            input_shape=[2, 16, 16, 4],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 4],
            expected_output_dtype='float32'
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        test_utils.layer_test(
            GGMSA,
            kwargs={'current_window': 6, 'pretrain_window': 4, 'num_heads': 2, 'qkv_bias': True, 'attn_drop': 0.,
                    'proj_drop': 0.},
            input_shape=[2, 16, 16, 4],
            input_dtype='float16',
            expected_output_shape=[None, 16, 16, 4],
            expected_output_dtype='float16'
        )


if __name__ == '__main__':
    tf.test.main()
