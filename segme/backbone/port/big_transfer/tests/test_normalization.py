import tensorflow as tf
from keras.testing_infra import test_combinations, test_utils
from keras.mixed_precision import policy as mixed_precision
from ..normalization import GroupNormalization


@test_combinations.run_all_keras_modes
class TestGroupNormalization(test_combinations.TestCase):
    def setUp(self):
        super(TestGroupNormalization, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestGroupNormalization, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            GroupNormalization,
            kwargs={},
            input_shape=[2, 48, 48, 64],
            input_dtype='float32',
            expected_output_shape=[None, 48, 48, 64],
            expected_output_dtype='float32'
        )

        mixed_precision.set_global_policy('mixed_float16')
        test_utils.layer_test(
            GroupNormalization,
            kwargs={},
            input_shape=[2, 48, 48, 64],
            input_dtype='float16',
            expected_output_shape=[None, 48, 48, 64],
            expected_output_dtype='float16'
        )


if __name__ == '__main__':
    tf.test.main()
