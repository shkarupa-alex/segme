import tensorflow as tf
from tensorflow.python.keras import keras_parameterized, testing_utils
from ..model import U2Net, U2NetP
from ....testing_utils import layer_multi_io_test


@keras_parameterized.run_all_keras_modes
class TestModel(keras_parameterized.TestCase):
    def test_u2net(self):
        layer_multi_io_test(
            U2Net,
            kwargs={'classes': 2},
            input_shapes=[(2, 64, 64, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[(None, 64, 64, 1)] * 7,
            expected_output_dtypes=['float32'] * 7
        )

    def test_u2netp(self):
        layer_multi_io_test(
            U2NetP,
            kwargs={'classes': 3},
            input_shapes=[(2, 64, 64, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[(None, 64, 64, 3)] * 7,
            expected_output_dtypes=['float32'] * 7
        )


if __name__ == '__main__':
    tf.test.main()
