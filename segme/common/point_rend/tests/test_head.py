import tensorflow as tf
from tensorflow.python.keras import keras_parameterized
from ..head import PointHead
from ....testing_utils import layer_multi_io_test


@keras_parameterized.run_all_keras_modes
class TestPointHead(keras_parameterized.TestCase):
    def setUp(self):
        super(TestPointHead, self).setUp()
        self.default_policy = tf.keras.mixed_precision.experimental.global_policy()

    def tearDown(self):
        super(TestPointHead, self).tearDown()
        tf.keras.mixed_precision.experimental.set_policy(self.default_policy)

    def test_layer(self):
        layer_multi_io_test(
            PointHead,
            kwargs={'classes': 5, 'units': [4, 3, 2], 'fines': 1, 'residual': False},
            input_shapes=[(2, 16, 5), (2, 16, 10)],
            input_dtypes=['float32', 'float32'],
            expected_output_shapes=[(None, 16, 5)],
            expected_output_dtypes=['float32']
        )
        layer_multi_io_test(
            PointHead,
            kwargs={'classes': 2, 'units': [4, 3, 2], 'fines': 2, 'residual': True},
            input_shapes=[(2, 16, 2), (2, 16, 10), (2, 16, 11)],
            input_dtypes=['float32', 'float32', 'float32'],
            expected_output_shapes=[(None, 16, 2)],
            expected_output_dtypes=['float32']
        )

        glob_policy = tf.keras.mixed_precision.experimental.global_policy()
        tf.keras.mixed_precision.experimental.set_policy('mixed_float16')
        layer_multi_io_test(
            PointHead,
            kwargs={'classes': 3, 'units': [4, 3, 2], 'fines': 1, 'residual': True},
            input_shapes=[(2, 16, 3), (2, 16, 10)],
            input_dtypes=['float16', 'float16'],
            expected_output_shapes=[(None, 16, 3)],
            expected_output_dtypes=['float16']
        )
        tf.keras.mixed_precision.experimental.set_policy(glob_policy)


if __name__ == '__main__':
    tf.test.main()
