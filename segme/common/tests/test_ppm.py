import tensorflow as tf
from keras.testing_infra import test_combinations, test_utils
from keras.mixed_precision import policy as mixed_precision
from segme.common.ppm import PyramidPooling


@test_combinations.run_all_keras_modes
class TestPyramidPooling(test_combinations.TestCase):
    def setUp(self):
        super(TestPyramidPooling, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestPyramidPooling, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            PyramidPooling,
            kwargs={'filters': 2, 'sizes': (1, 2, 3, 6)},
            input_shape=[2, 18, 18, 3],
            input_dtype='float32',
            expected_output_shape=[None, 18, 18, 2],
            expected_output_dtype='float32'
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        test_utils.layer_test(
            PyramidPooling,
            kwargs={'filters': 32, 'sizes': (1, 2, 3, 6)},
            input_shape=[2, 18, 18, 64],
            input_dtype='float16',
            expected_output_shape=[None, 18, 18, 32],
            expected_output_dtype='float16'
        )


if __name__ == '__main__':
    tf.test.main()
