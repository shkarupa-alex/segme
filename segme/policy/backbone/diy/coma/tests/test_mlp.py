import tensorflow as tf
from keras.mixed_precision import policy as mixed_precision
from keras.testing_infra import test_combinations, test_utils
from segme.policy.backbone.diy.coma.mlp import MLP


@test_combinations.run_all_keras_modes
class TestMLP(test_combinations.TestCase):
    def setUp(self):
        super(TestMLP, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestMLP, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            MLP,
            kwargs={'ratio': 0.5, 'dropout': 0.},
            input_shape=[2, 4, 4, 3],
            input_dtype='float32',
            expected_output_shape=[None, 4, 4, 3],
            expected_output_dtype='float32'
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        test_utils.layer_test(
            MLP,
            kwargs={'ratio': 1.5, 'dropout': 0.2},
            input_shape=[2, 4, 4, 3],
            input_dtype='float16',
            expected_output_shape=[None, 4, 4, 3],
            expected_output_dtype='float16'
        )


if __name__ == '__main__':
    tf.test.main()
