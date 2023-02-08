import tensorflow as tf
from keras.mixed_precision import policy as mixed_precision
from keras.testing_infra import test_combinations
from segme.common.carafe import CarafeConvolution
from segme.testing_utils import layer_multi_io_test


@test_combinations.run_all_keras_modes
class TestCarafeConvolution(test_combinations.TestCase):
    def setUp(self):
        super(TestCarafeConvolution, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestCarafeConvolution, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        layer_multi_io_test(
            CarafeConvolution,
            kwargs={'kernel_size': 3},
            input_shapes=[(2, 3, 4, 6), (2, 6, 8, 18)],
            input_dtypes=['float32'] * 2,
            expected_output_shapes=[(None, 6, 8, 6)],
            expected_output_dtypes=['float32']
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        layer_multi_io_test(
            CarafeConvolution,
            kwargs={'kernel_size': 3},
            input_shapes=[(2, 3, 4, 6), (2, 6, 8, 18)],
            input_dtypes=['float16'] * 2,
            expected_output_shapes=[(None, 6, 8, 6)],
            expected_output_dtypes=['float16']
        )


if __name__ == '__main__':
    tf.test.main()
