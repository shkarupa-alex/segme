import tensorflow as tf
from keras import keras_parameterized
from ..cfm import CFM
from ....testing_utils import layer_multi_io_test


@keras_parameterized.run_all_keras_modes
class TestCFM(keras_parameterized.TestCase):
    def test_layer(self):
        layer_multi_io_test(
            CFM,
            kwargs={'filters': 5},
            input_shapes=[(2, 16, 16, 3), (2, 4, 4, 3)],
            input_dtypes=['float32', 'float32'],
            expected_output_shapes=[(None, 16, 16, 5), (None, 16, 16, 5)],
            expected_output_dtypes=['float32', 'float32']
        )


if __name__ == '__main__':
    tf.test.main()
