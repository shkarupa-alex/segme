import numpy as np
import tensorflow as tf
from tensorflow.python.keras import keras_parameterized, testing_utils
from ..tochannel import ToChannelFirst, ToChannelLast, to_channel_last, to_channel_first


@keras_parameterized.run_all_keras_modes
class TestToChannel(keras_parameterized.TestCase):
    def test_layer(self):
        testing_utils.layer_test(
            ToChannelLast,
            kwargs={},
            input_shape=[2, 10, 16, 16],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 10],
            expected_output_dtype='float32'
        )
        testing_utils.layer_test(
            ToChannelFirst,
            kwargs={},
            input_shape=[2, 16, 16, 10],
            input_dtype='float32',
            expected_output_shape=[None, 10, 16, 16],
            expected_output_dtype='float32'
        )

    def test_invert(self):
        source = np.random.rand(2, 16, 16, 3)
        result = to_channel_last(source)
        result = to_channel_first(result)
        result = self.evaluate(result)

        self.assertAllClose(source, result)


if __name__ == '__main__':
    tf.test.main()
