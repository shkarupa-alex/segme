import tensorflow as tf
from keras import keras_parameterized
from ..aim import AIM
from ....testing_utils import layer_multi_io_test


@keras_parameterized.run_all_keras_modes
class TestAIM(keras_parameterized.TestCase):
    def test_layer(self):
        layer_multi_io_test(
            AIM,
            kwargs={'filters': [5, 4, 3, 2, 1]},
            input_shapes=[(2, 32, 32, 1), (2, 16, 16, 2), (2, 8, 8, 3), (2, 4, 4, 4), (2, 2, 2, 5)],
            input_dtypes=['float32'] * 5,
            expected_output_shapes=[
                (None, 32, 32, 5), (None, 16, 16, 4), (None, 8, 8, 3), (None, 4, 4, 2), (None, 2, 2, 1)],
            expected_output_dtypes=['float32'] * 5
        )


if __name__ == '__main__':
    tf.test.main()
