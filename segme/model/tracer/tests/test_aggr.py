import tensorflow as tf
from keras import keras_parameterized
from ..aggr import Aggregation
from ....testing_utils import layer_multi_io_test


@keras_parameterized.run_all_keras_modes
class TestAggregation(keras_parameterized.TestCase):
    def test_layer(self):
        layer_multi_io_test(
            Aggregation,
            kwargs={'confidence': 0.1},
            input_shapes=[(2, 32, 32, 16), (2, 16, 16, 32), (2, 8, 8, 64)],
            input_dtypes=['float32'] * 3,
            expected_output_shapes=[(None, 32, 32, 1)],
            expected_output_dtypes=['float32']
        )


if __name__ == '__main__':
    tf.test.main()
