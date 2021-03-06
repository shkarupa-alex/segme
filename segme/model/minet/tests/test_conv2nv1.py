import tensorflow as tf
from tensorflow.python.keras import keras_parameterized
from ..conv2nv1 import Conv2nV1
from ....testing_utils import layer_multi_io_test


@keras_parameterized.run_all_keras_modes
class TestConv2nV1(keras_parameterized.TestCase):
    def test_layer(self):
        layer_multi_io_test(
            Conv2nV1,
            kwargs={'filters': 7, 'main': 0},
            input_shapes=[(2, 16, 16, 2), (2, 8, 8, 4)],
            input_dtypes=['float32', 'float32'],
            expected_output_shapes=[(None, 16, 16, 7)],
            expected_output_dtypes=['float32']
        )
        layer_multi_io_test(
            Conv2nV1,
            kwargs={'filters': 7, 'main': 1},
            input_shapes=[(2, 16, 16, 2), (2, 8, 8, 4)],
            input_dtypes=['float32', 'float32'],
            expected_output_shapes=[(None, 8, 8, 7)],
            expected_output_dtypes=['float32']
        )


if __name__ == '__main__':
    tf.test.main()
