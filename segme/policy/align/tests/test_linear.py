import tensorflow as tf
from keras.mixed_precision import policy as mixed_precision
from keras.testing_infra import test_combinations, test_utils
from segme.policy.align.linear import BilinearFeatureAlignment
from segme.testing_utils import layer_multi_io_test


@test_combinations.run_all_keras_modes
class TestBilinearFeatureAlignment(test_combinations.TestCase):
    def setUp(self):
        super(TestBilinearFeatureAlignment, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestBilinearFeatureAlignment, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        layer_multi_io_test(
            BilinearFeatureAlignment,
            kwargs={'filters': 6},
            input_shapes=[(2, 16, 16, 4), (2, 8, 8, 8)],
            input_dtypes=['float32'] * 2,
            expected_output_shapes=[(None, 16, 16, 6)],
            expected_output_dtypes=['float32']
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        layer_multi_io_test(
            BilinearFeatureAlignment,
            kwargs={'filters': 6},
            input_shapes=[(2, 16, 16, 4), (2, 8, 8, 8)],
            input_dtypes=['float16'] * 2,
            expected_output_shapes=[(None, 16, 16, 6)],
            expected_output_dtypes=['float16']
        )


if __name__ == '__main__':
    tf.test.main()
