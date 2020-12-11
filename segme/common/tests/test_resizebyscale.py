import numpy as np
import tensorflow as tf
from tensorflow.python.keras import keras_parameterized, testing_utils
from ..resizebyscale import ResizeByScale, resize_by_scale


@keras_parameterized.run_all_keras_modes
class TestResizeByScale(keras_parameterized.TestCase):
    def setUp(self):
        super(TestResizeByScale, self).setUp()
        self.default_policy = tf.keras.mixed_precision.experimental.global_policy()

    def tearDown(self):
        super(TestResizeByScale, self).tearDown()
        tf.keras.mixed_precision.experimental.set_policy(self.default_policy)

    def test_layer(self):
        testing_utils.layer_test(
            ResizeByScale,
            kwargs={'scale': 2},
            input_shape=(2, 16, 16, 10),
            input_dtype='float32',
            expected_output_shape=(None, 32, 32, 10),
            expected_output_dtype='float32'
        )

        glob_policy = tf.keras.mixed_precision.experimental.global_policy()
        tf.keras.mixed_precision.experimental.set_policy('mixed_float16')
        testing_utils.layer_test(
            ResizeByScale,
            kwargs={'scale': 2},
            input_shape=(2, 16, 16, 10),
            input_dtype='float16',
            expected_output_shape=(None, 32, 32, 10),
            expected_output_dtype='float32'
        )
        tf.keras.mixed_precision.experimental.set_policy(glob_policy)

    def test_shortcut(self):
        resize_by_scale(np.random.rand(2, 16, 16, 3), scale=2, align_corners=False)


if __name__ == '__main__':
    tf.test.main()
