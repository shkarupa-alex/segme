import tensorflow as tf
from tf_keras import mixed_precision
from tf_keras.src.testing_infra import test_combinations, test_utils
from segme.common.gavg import GlobalAverage


@test_combinations.run_all_keras_modes
class TestGlobalAverage(test_combinations.TestCase):
    def setUp(self):
        super(TestGlobalAverage, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestGlobalAverage, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            GlobalAverage,
            kwargs={},
            input_shape=[2, 36, 36, 3],
            input_dtype='float32',
            expected_output_shape=[None, 36, 36, 3],
            expected_output_dtype='float32'
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        test_utils.layer_test(
            GlobalAverage,
            kwargs={},
            input_shape=[2, 9, 9, 32],
            input_dtype='float16',
            expected_output_shape=[None, 9, 9, 32],
            expected_output_dtype='float16'
        )


if __name__ == '__main__':
    tf.test.main()
