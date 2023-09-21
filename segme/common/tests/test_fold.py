import tensorflow as tf
from keras import mixed_precision
from keras.src.testing_infra import test_combinations, test_utils
from segme.common.fold import Fold, UnFold


@test_combinations.run_all_keras_modes
class TestFold(test_combinations.TestCase):
    def setUp(self):
        super(TestFold, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestFold, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            Fold,
            kwargs={'size': 2},
            input_shape=[2, 8, 8, 4],
            input_dtype='float32',
            expected_output_shape=[None, 4, 4, 16],
            expected_output_dtype='float32'
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        test_utils.layer_test(
            Fold,
            kwargs={'size': 4},
            input_shape=[2, 32, 32, 9],
            input_dtype='float16',
            expected_output_shape=[None, 8, 8, 144],
            expected_output_dtype='float16'
        )


@test_combinations.run_all_keras_modes
class TestUnFold(test_combinations.TestCase):
    def setUp(self):
        super(TestUnFold, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestUnFold, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            UnFold,
            kwargs={'size': 2},
            input_shape=[2, 8, 8, 4],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 1],
            expected_output_dtype='float32'
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        test_utils.layer_test(
            UnFold,
            kwargs={'size': 4},
            input_shape=[2, 9, 9, 32],
            input_dtype='float16',
            expected_output_shape=[None, 36, 36, 2],
            expected_output_dtype='float16'
        )


if __name__ == '__main__':
    tf.test.main()
