import tensorflow as tf
from tf_keras import mixed_precision
from tf_keras.src.testing_infra import test_combinations, test_utils
from segme.common.simpool import SimPool


@test_combinations.run_all_keras_modes
class TestSimPool(test_combinations.TestCase):
    def setUp(self):
        super(TestSimPool, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestSimPool, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            SimPool,
            kwargs={'num_heads': 1, 'qkv_bias': True},
            input_shape=[2, 16, 16, 4],
            input_dtype='float32',
            expected_output_shape=[None, 4],
            expected_output_dtype='float32'
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        test_utils.layer_test(
            SimPool,
            kwargs={'num_heads': 4, 'qkv_bias': False},
            input_shape=[2, 16, 16, 4],
            input_dtype='float16',
            expected_output_shape=[None, 4],
            expected_output_dtype='float16'
        )


if __name__ == '__main__':
    tf.test.main()
