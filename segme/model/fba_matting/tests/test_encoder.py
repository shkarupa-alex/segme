import tensorflow as tf
from keras import keras_parameterized
from ..encoder import Encoder
from ....testing_utils import layer_multi_io_test


@keras_parameterized.run_all_keras_modes
class TestEncoder(keras_parameterized.TestCase):
    def test_layer(self):
        layer_multi_io_test(
            Encoder,
            kwargs={'bone_arch': 'bit_m_r50x1_stride_8', 'bone_init': 'imagenet'},
            input_shapes=[(2, 512, 512, 11)],
            input_dtypes=['uint8'],
            expected_output_shapes=[(None, 256, 256, 64), (None, 128, 128, 256), (None, 64, 64, 2048)],
            expected_output_dtypes=['float32'] * 3
        )


if __name__ == '__main__':
    tf.test.main()
