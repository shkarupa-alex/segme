import tensorflow as tf
from keras.testing_infra import test_combinations
from ..dense import DenseBlock
from ....testing_utils import layer_multi_io_test


@test_combinations.run_all_keras_modes
class TestDenseBlock(test_combinations.TestCase):
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
