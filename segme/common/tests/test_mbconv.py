import tensorflow as tf
from keras.mixed_precision import policy as mixed_precision
from keras.testing_infra import test_combinations, test_utils
from segme.common.mbconv import MBConv


@test_combinations.run_all_keras_modes
class TestMBConv(test_combinations.TestCase):
    def setUp(self):
        super(TestMBConv, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestMBConv, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            MBConv,
            kwargs={'filters': 4, 'kernel_size': 3, 'fused': True, 'strides': 1, 'expand_ratio': 4., 'se_ratio': 0.,
                    'gamma_initializer': 'ones', 'drop_ratio': 0.},
            input_shape=[2, 8, 8, 3],
            input_dtype='float32',
            expected_output_shape=[None, 8, 8, 4],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            MBConv,
            kwargs={'filters': 4, 'kernel_size': 3, 'fused': False, 'strides': 1, 'expand_ratio': 4., 'se_ratio': 0.,
                    'gamma_initializer': 'ones', 'drop_ratio': 0.},
            input_shape=[2, 8, 8, 3],
            input_dtype='float32',
            expected_output_shape=[None, 8, 8, 4],
            expected_output_dtype='float32'
        )

        test_utils.layer_test(
            MBConv,
            kwargs={'filters': 4, 'kernel_size': 3, 'fused': True, 'strides': 2, 'expand_ratio': 4., 'se_ratio': 0.,
                    'gamma_initializer': 'ones', 'drop_ratio': 0.},
            input_shape=[2, 8, 8, 3],
            input_dtype='float32',
            expected_output_shape=[None, 4, 4, 4],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            MBConv,
            kwargs={'filters': 4, 'kernel_size': 3, 'fused': False, 'strides': 2, 'expand_ratio': 4., 'se_ratio': 0.,
                    'gamma_initializer': 'ones', 'drop_ratio': 0.},
            input_shape=[2, 8, 8, 3],
            input_dtype='float32',
            expected_output_shape=[None, 4, 4, 4],
            expected_output_dtype='float32'
        )

        test_utils.layer_test(
            MBConv,
            kwargs={'filters': 4, 'kernel_size': 3, 'fused': True, 'strides': 1, 'expand_ratio': 4., 'se_ratio': 0.2,
                    'gamma_initializer': 'ones', 'drop_ratio': 0.},
            input_shape=[2, 8, 8, 3],
            input_dtype='float32',
            expected_output_shape=[None, 8, 8, 4],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            MBConv,
            kwargs={'filters': 4, 'kernel_size': 3, 'fused': False, 'strides': 1, 'expand_ratio': 4., 'se_ratio': 0.2,
                    'gamma_initializer': 'ones', 'drop_ratio': 0.},
            input_shape=[2, 8, 8, 3],
            input_dtype='float32',
            expected_output_shape=[None, 8, 8, 4],
            expected_output_dtype='float32'
        )

        test_utils.layer_test(
            MBConv,
            kwargs={'filters': 4, 'kernel_size': 3, 'fused': True, 'strides': 1, 'expand_ratio': 4., 'se_ratio': 0.,
                    'gamma_initializer': 'zeros', 'drop_ratio': 0.},
            input_shape=[2, 8, 8, 4],
            input_dtype='float32',
            expected_output_shape=[None, 8, 8, 4],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            MBConv,
            kwargs={'filters': 4, 'kernel_size': 3, 'fused': False, 'strides': 1, 'expand_ratio': 4., 'se_ratio': 0.,
                    'gamma_initializer': 'zeros', 'drop_ratio': 0.},
            input_shape=[2, 8, 8, 4],
            input_dtype='float32',
            expected_output_shape=[None, 8, 8, 4],
            expected_output_dtype='float32'
        )

        test_utils.layer_test(
            MBConv,
            kwargs={'filters': 4, 'kernel_size': 3, 'fused': True, 'strides': 1, 'expand_ratio': 4., 'se_ratio': 0.,
                    'gamma_initializer': 'ones', 'drop_ratio': 0.2},
            input_shape=[2, 8, 8, 4],
            input_dtype='float32',
            expected_output_shape=[None, 8, 8, 4],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            MBConv,
            kwargs={'filters': 4, 'kernel_size': 3, 'fused': False, 'strides': 1, 'expand_ratio': 4., 'se_ratio': 0.,
                    'gamma_initializer': 'ones', 'drop_ratio': 0.2},
            input_shape=[2, 8, 8, 4],
            input_dtype='float32',
            expected_output_shape=[None, 8, 8, 4],
            expected_output_dtype='float32'
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        test_utils.layer_test(
            MBConv,
            kwargs={'filters': 4, 'kernel_size': 3, 'fused': True, 'strides': 1, 'expand_ratio': 4., 'se_ratio': 0.2,
                    'gamma_initializer': 'ones', 'drop_ratio': 0.2},
            input_shape=[2, 8, 8, 4],
            input_dtype='float16',
            expected_output_shape=[None, 8, 8, 4],
            expected_output_dtype='float16'
        )
        test_utils.layer_test(
            MBConv,
            kwargs={'filters': 4, 'kernel_size': 3, 'fused': False, 'strides': 1, 'expand_ratio': 4., 'se_ratio': 0.2,
                    'gamma_initializer': 'zeros', 'drop_ratio': 0.2},
            input_shape=[2, 8, 8, 4],
            input_dtype='float16',
            expected_output_shape=[None, 8, 8, 4],
            expected_output_dtype='float16'
        )


if __name__ == '__main__':
    tf.test.main()
