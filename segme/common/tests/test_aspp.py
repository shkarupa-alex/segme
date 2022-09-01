import tensorflow as tf
from keras.testing_infra import test_combinations, test_utils
from keras.mixed_precision import policy as mixed_precision
from segme.common.aspp import AtrousSpatialPyramidPooling


@test_combinations.run_all_keras_modes
class TestAtrousSpatialPyramidPooling(test_combinations.TestCase):
    def setUp(self):
        super(TestAtrousSpatialPyramidPooling, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestAtrousSpatialPyramidPooling, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            AtrousSpatialPyramidPooling,
            kwargs={'filters': 10, 'stride': 8},
            input_shape=[2, 36, 36, 3],
            input_dtype='float32',
            expected_output_shape=[None, 36, 36, 10],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            AtrousSpatialPyramidPooling,
            kwargs={'filters': 64, 'stride': 16},
            input_shape=[2, 18, 18, 32],
            input_dtype='float32',
            expected_output_shape=[None, 18, 18, 64],
            expected_output_dtype='float32'
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        test_utils.layer_test(
            AtrousSpatialPyramidPooling,
            kwargs={'filters': 64, 'stride': 32},
            input_shape=[2, 9, 9, 32],
            input_dtype='float16',
            expected_output_shape=[None, 9, 9, 64],
            expected_output_dtype='float16'
        )


if __name__ == '__main__':
    tf.test.main()
