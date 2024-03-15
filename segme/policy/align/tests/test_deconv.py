import tensorflow as tf
from tf_keras import mixed_precision
from tf_keras.src.testing_infra import test_combinations, test_utils
from segme.policy.align.deconv import DeconvolutionFeatureAlignment
from segme.testing_utils import layer_multi_io_test


@test_combinations.run_all_keras_modes
class TestDeconvolutionFeatureAlignment(test_combinations.TestCase):
    def setUp(self):
        super(TestDeconvolutionFeatureAlignment, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestDeconvolutionFeatureAlignment, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        layer_multi_io_test(
            DeconvolutionFeatureAlignment,
            kwargs={'filters': 6, 'kernel_size': 4},
            input_shapes=[(2, 16, 16, 4), (2, 8, 8, 8)],
            input_dtypes=['float32'] * 2,
            expected_output_shapes=[(None, 16, 16, 6)],
            expected_output_dtypes=['float32']
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        layer_multi_io_test(
            DeconvolutionFeatureAlignment,
            kwargs={'filters': 6},
            input_shapes=[(2, 16, 16, 4), (2, 8, 8, 8)],
            input_dtypes=['float16'] * 2,
            expected_output_shapes=[(None, 16, 16, 6)],
            expected_output_dtypes=['float16']
        )


if __name__ == '__main__':
    tf.test.main()
