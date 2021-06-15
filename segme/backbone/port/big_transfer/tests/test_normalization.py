import numpy as np
import tensorflow as tf
from tensorflow.python.keras import keras_parameterized, testing_utils
from ..normalization import GroupNormalization


@keras_parameterized.run_all_keras_modes
class TestGroupNormalization(keras_parameterized.TestCase):
    def setUp(self):
        super(TestGroupNormalization, self).setUp()
        self.default_policy = tf.keras.mixed_precision.experimental.global_policy()

    def tearDown(self):
        super(TestGroupNormalization, self).tearDown()
        tf.keras.mixed_precision.experimental.set_policy(self.default_policy)

    def test_layer(self):
        testing_utils.layer_test(
            GroupNormalization,
            kwargs={},
            input_shape=[2, 48, 48, 64],
            input_dtype='float32',
            expected_output_shape=[None, 48, 48, 64],
            expected_output_dtype='float32'
        )

        tf.keras.mixed_precision.experimental.set_policy('mixed_float16')
        testing_utils.layer_test(
            GroupNormalization,
            kwargs={},
            input_shape=[2, 48, 48, 64],
            input_dtype='float16',
            expected_output_shape=[None, 48, 48, 64],
            expected_output_dtype='float16'
        )


if __name__ == '__main__':
    tf.test.main()
