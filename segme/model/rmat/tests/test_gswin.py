import tensorflow as tf
from keras import keras_parameterized
from ..gswin import GidedSwin
from ....testing_utils import layer_multi_io_test


@keras_parameterized.run_all_keras_modes
class TestGidedSwin(keras_parameterized.TestCase):
    def test_layer(self):
        layer_multi_io_test(
            GidedSwin,
            kwargs={'arch': 'tiny_224'},
            input_shapes=[(2, 224, 224, 3), (2, 224, 224, 6)],
            input_dtypes=['uint8'] * 2,
            expected_output_shapes=[(None, 56, 56, 96), (None, 28, 28, 192), (None, 14, 14, 384), (None, 7, 7, 768)],
            expected_output_dtypes=['float32'] * 4
        )


if __name__ == '__main__':
    tf.test.main()
