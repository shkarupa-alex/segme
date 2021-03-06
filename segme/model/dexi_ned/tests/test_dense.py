import tensorflow as tf
from tensorflow.python.keras import keras_parameterized
from ..dense import DenseBlock
from ....testing_utils import layer_multi_io_test


@keras_parameterized.run_all_keras_modes
class TestDenseBlock(keras_parameterized.TestCase):
    def test_layer(self):
        layer_multi_io_test(
            DenseBlock,
            kwargs={'num_layers': 2, 'out_features': 7},
            input_shapes=[(2, 16, 16, 5), (2, 16, 16, 7)],
            input_dtypes=['float32', 'float32'],
            expected_output_shapes=[(None, 16, 16, 7)],
            expected_output_dtypes=['float32']
        )


if __name__ == '__main__':
    tf.test.main()
