import tensorflow as tf
from tensorflow.python.keras import keras_parameterized
from ..dense import DexiNedDenseBlock
from ....testing_utils import layer_multi_io_test


@keras_parameterized.run_all_keras_modes
class TestDexiNedDenseBlock(keras_parameterized.TestCase):
    def test_layer(self):
        layer_multi_io_test(
            DexiNedDenseBlock,
            kwargs={'num_layers': 2, 'out_features': 5},
            input_shapes=[(2, 16, 16, 5), (2, 16, 16, 5)],
            input_dtypes=['float32', 'float32'],
            expected_output_dtypes=['float32', 'float32']
        )


if __name__ == '__main__':
    tf.test.main()
