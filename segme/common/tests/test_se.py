import tensorflow as tf
from keras.mixed_precision import policy as mixed_precision
from keras.testing_infra import test_combinations, test_utils
from segme.common.se import SE


@test_combinations.run_all_keras_modes
class TestSE(test_combinations.TestCase):
    def setUp(self):
        super(TestSE, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestSE, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            SE,
            kwargs={'ratio': 0.5},
            input_shape=[2, 4, 4, 3],
            input_dtype='float32',
            expected_output_shape=[None, 4, 4, 3],
            expected_output_dtype='float32'
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        test_utils.layer_test(
            SE,
            kwargs={'ratio': 0.25},
            input_shape=[2, 4, 4, 4],
            input_dtype='float16',
            expected_output_shape=[None, 4, 4, 4],
            expected_output_dtype='float16'
        )


if __name__ == '__main__':
    tf.test.main()
