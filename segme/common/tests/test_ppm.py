import tensorflow as tf
from keras import keras_parameterized, testing_utils
from keras.mixed_precision import policy as mixed_precision
from ..ppm import PyramidPooling


@keras_parameterized.run_all_keras_modes
class TestPyramidPooling(keras_parameterized.TestCase):
    def setUp(self):
        super(TestPyramidPooling, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestPyramidPooling, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        testing_utils.layer_test(
            PyramidPooling,
            kwargs={'filters': 2, 'sizes': (1, 2, 3, 6)},
            input_shape=[2, 18, 18, 3],
            input_dtype='float32',
            expected_output_shape=[None, 18, 18, 2],
            expected_output_dtype='float32'
        )

        mixed_precision.set_global_policy('mixed_float16')
        testing_utils.layer_test(
            PyramidPooling,
            kwargs={'filters': 32, 'sizes': (1, 2, 3, 6), 'activation': 'leaky_relu', 'standardized': True},
            input_shape=[2, 18, 18, 64],
            input_dtype='float16',
            expected_output_shape=[None, 18, 18, 32],
            expected_output_dtype='float16'
        )


if __name__ == '__main__':
    tf.test.main()
