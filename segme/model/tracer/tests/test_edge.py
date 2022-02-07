import tensorflow as tf
from keras import keras_parameterized
from ..edge import FrequencyEdge
from ....testing_utils import layer_multi_io_test


@keras_parameterized.run_all_keras_modes
class TestFrequencyEdge(keras_parameterized.TestCase):
    def test_layer(self):
        layer_multi_io_test(
            FrequencyEdge,
            kwargs={'radius': 5, 'confidence': 0.1},
            input_shapes=[(2, 32, 32, 16)],
            input_dtypes=['float32'],
            expected_output_shapes=[(None, 32, 32, 16), (None, 32, 32, 1)],
            expected_output_dtypes=['float32'] * 2
        )


if __name__ == '__main__':
    tf.test.main()
