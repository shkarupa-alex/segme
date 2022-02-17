import numpy as np
import tensorflow as tf
from keras import keras_parameterized
from ..edge import FrequencyEdge, extract_edges
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


@keras_parameterized.run_all_keras_modes
class TestExtractEdges(keras_parameterized.TestCase):
    def test_value(self):
        targets = np.array(
            [0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1],
            'int32').reshape((4, 8))

        gy, gx = np.gradient(targets)
        expected = gy * gy + gx * gx
        expected[expected != 0] = 1
        expected = expected.astype('int32')

        result = extract_edges(targets[None, ..., None])
        result = self.evaluate(result)[0, ..., 0]

        self.assertTrue(np.all(expected == result))


if __name__ == '__main__':
    tf.test.main()
