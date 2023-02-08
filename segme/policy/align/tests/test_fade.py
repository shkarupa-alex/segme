import tensorflow as tf
from keras.mixed_precision import policy as mixed_precision
from keras.testing_infra import test_combinations, test_utils
from segme.policy.align.fade import FadeFeatureAlignment, SemiShift
from segme.testing_utils import layer_multi_io_test


@test_combinations.run_all_keras_modes
class TestFadeFeatureAlignment(test_combinations.TestCase):
    def setUp(self):
        super(TestFadeFeatureAlignment, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestFadeFeatureAlignment, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        layer_multi_io_test(
            FadeFeatureAlignment,
            kwargs={'filters': 6, 'kernel_size': 5, 'embedding_size': 16},
            input_shapes=[(2, 16, 16, 4), (2, 8, 8, 8)],
            input_dtypes=['float32'] * 2,
            expected_output_shapes=[(None, 16, 16, 6)],
            expected_output_dtypes=['float32']
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        layer_multi_io_test(
            FadeFeatureAlignment,
            kwargs={'filters': 6, 'kernel_size': 5, 'embedding_size': 16},
            input_shapes=[(2, 16, 16, 4), (2, 8, 8, 8)],
            input_dtypes=['float16'] * 2,
            expected_output_shapes=[(None, 16, 16, 6)],
            expected_output_dtypes=['float16']
        )


@test_combinations.run_all_keras_modes
class TestSemiShift(test_combinations.TestCase):
    def setUp(self):
        super(TestSemiShift, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestSemiShift, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        layer_multi_io_test(
            SemiShift,
            kwargs={'filters': 25, 'kernel_size': 3, 'embedding_size': 8},
            input_shapes=[(2, 6, 8, 12), (2, 3, 4, 6)],
            input_dtypes=['float32'] * 2,
            expected_output_shapes=[(None, 6, 8, 25)],
            expected_output_dtypes=['float32']
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        layer_multi_io_test(
            SemiShift,
            kwargs={'filters': 25, 'kernel_size': 3, 'embedding_size': 8},
            input_shapes=[(2, 6, 8, 12), (2, 3, 4, 6)],
            input_dtypes=['float16'] * 2,
            expected_output_shapes=[(None, 6, 8, 25)],
            expected_output_dtypes=['float16']
        )


if __name__ == '__main__':
    tf.test.main()
