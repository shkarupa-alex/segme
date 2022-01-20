import tensorflow as tf
from keras import keras_parameterized
from ..encoder import Encoder
from ....testing_utils import layer_multi_io_test


@keras_parameterized.run_all_keras_modes
class TestEncoder(keras_parameterized.TestCase):
    def test_layer(self):
        layer_multi_io_test(
            Encoder,
            kwargs={
                'bone_arch': 'resnet_50', 'bone_init': 'imagenet', 'bone_train': False,
                'aspp_filters': 10, 'aspp_stride': 16},
            input_shapes=[(2, 224, 224, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[
                (None, 56, 56, 256),
                (None, 14, 14, 10)
            ],
            expected_output_dtypes=['float32', 'float32']
        )
        layer_multi_io_test(
            Encoder,
            kwargs={
                'bone_arch': 'resnet_50', 'bone_init': 'imagenet', 'bone_train': False,
                'aspp_filters': 10, 'aspp_stride': 16, 'add_strides': [4, 8, 2]},
            input_shapes=[(2, 224, 224, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[
                (None, 56, 56, 256),
                (None, 14, 14, 10),
                (None, 56, 56, 256),
                (None, 28, 28, 512),
                (None, 112, 112, 64)
            ],
            expected_output_dtypes=['float32', 'float32', 'float32', 'float32', 'float32']
        )


if __name__ == '__main__':
    tf.test.main()
