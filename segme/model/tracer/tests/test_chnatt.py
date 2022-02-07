import tensorflow as tf
from keras import keras_parameterized
from ..chnatt import ChannelAttention
from ....testing_utils import layer_multi_io_test


@keras_parameterized.run_all_keras_modes
class TestChannelAttention(keras_parameterized.TestCase):
    def test_layer(self):
        layer_multi_io_test(
            ChannelAttention,
            kwargs={'confidence': 0.1},
            input_shapes=[(2, 32, 32, 16)],
            input_dtypes=['float32'],
            expected_output_shapes=[(None, 32, 32, 16), (None, 1, 1, 16)],
            expected_output_dtypes=['float32'] * 2
        )


if __name__ == '__main__':
    tf.test.main()
