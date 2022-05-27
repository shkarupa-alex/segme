import numpy as np
import tensorflow as tf
from keras.testing_infra import test_combinations, test_utils
from keras.mixed_precision import policy as mixed_precision
from ..resizebyscale import ResizeByScale, resize_by_scale


@test_combinations.run_all_keras_modes
class TestResizeByScale(test_combinations.TestCase):
    def setUp(self):
        super(TestResizeByScale, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestResizeByScale, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            ResizeByScale,
            kwargs={'scale': 2},
            input_shape=(2, 16, 16, 10),
            input_dtype='float32',
            expected_output_shape=(None, 32, 32, 10),
            expected_output_dtype='float32'
        )

        mixed_precision.set_global_policy('mixed_float16')
        test_utils.layer_test(
            ResizeByScale,
            kwargs={'scale': 2},
            input_shape=(2, 16, 16, 10),
            input_dtype='float16',
            expected_output_shape=(None, 32, 32, 10),
            expected_output_dtype='float16'
        )

    def test_shortcut(self):
        resize_by_scale(np.random.rand(2, 16, 16, 3), scale=2, antialias=True)


if __name__ == '__main__':
    tf.test.main()
