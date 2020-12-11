import tensorflow as tf
from tensorflow.python.keras import keras_parameterized
from ..rend import PointRend
from ....testing_utils import layer_multi_io_test


@keras_parameterized.run_all_keras_modes
class TestPointRend(keras_parameterized.TestCase):
    def setUp(self):
        super(TestPointRend, self).setUp()
        self.default_policy = tf.keras.mixed_precision.experimental.global_policy()

    def tearDown(self):
        super(TestPointRend, self).tearDown()
        tf.keras.mixed_precision.experimental.set_policy(self.default_policy)

    def test_layer(self):
        layer_multi_io_test(
            PointRend,
            kwargs={
                'classes': 5, 'units': [2, 3], 'points': (40, 10), 'oversample': 3, 'importance': 0.75, 'fines': 1,
                'residual': False, 'align_corners': True},
            input_shapes=[(4, 64, 64, 3), (4, 16, 16, 5), (4, 32, 32, 6)],
            input_dtypes=['float32', 'float32', 'float32'],
            expected_output_shapes=[(None, 64, 64, 5), (None, None, 5), (None, None, 2)],
            expected_output_dtypes=['float32', 'float32', 'float32']
        )

        glob_policy = tf.keras.mixed_precision.experimental.global_policy()
        tf.keras.mixed_precision.experimental.set_policy('mixed_float16')
        layer_multi_io_test(
            PointRend,
            kwargs={
                'classes': 5, 'units': [4], 'points': (40, 10), 'oversample': 4, 'importance': 0.95, 'fines': 2,
                'residual': True, 'align_corners': False},
            input_shapes=[(4, 64, 64, 3), (4, 16, 16, 5), (4, 48, 48, 6), (4, 32, 32, 7)],
            input_dtypes=['uint8', 'float16', 'float16', 'float16'],
            expected_output_shapes=[(None, 64, 64, 5), (None, None, 5), (None, None, 2)],
            expected_output_dtypes=['float32', 'float32', 'float32']
        )
        tf.keras.mixed_precision.experimental.set_policy(glob_policy)


if __name__ == '__main__':
    tf.test.main()
