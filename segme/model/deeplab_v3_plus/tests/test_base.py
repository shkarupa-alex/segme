import numpy as np
import tensorflow as tf
from keras import keras_parameterized, testing_utils
from keras.mixed_precision import policy as mixed_precision
from tensorflow.python.training.tracking import util as trackable_util
from tensorflow.python.util import object_identity
from ..base import DeepLabV3PlusBase
from ....testing_utils import layer_multi_io_test


@keras_parameterized.run_all_keras_modes
class TestDeepLabV3PlusBase(keras_parameterized.TestCase):
    def setUp(self):
        super(TestDeepLabV3PlusBase, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestDeepLabV3PlusBase, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        # TODO: wait for issue with Sequential model restoring
        #  will be resolved to migrate back on testing_utils.layer_test
        layer_multi_io_test(
            DeepLabV3PlusBase,
            kwargs={
                'classes': 4, 'bone_arch': 'resnet_50', 'bone_init': 'imagenet', 'bone_train': False,
                'aspp_filters': 8, 'aspp_stride': 32, 'low_filters': 16, 'decoder_filters': 5, 'add_strides': (2, 4)},
            input_shapes=[(2, 224, 224, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[(None, 56, 56, 4), (None, 56, 56, 5), (None, 112, 112, 64), (None, 56, 56, 256)],
            expected_output_dtypes=['float32'] * 4
        )

        mixed_precision.set_global_policy('mixed_float16')
        layer_multi_io_test(
            DeepLabV3PlusBase,
            kwargs={
                'classes': 1, 'bone_arch': 'resnet_50', 'bone_init': 'imagenet', 'bone_train': False,
                'aspp_filters': 8, 'aspp_stride': 32, 'low_filters': 16, 'decoder_filters': 4, 'add_strides': None},
            input_shapes=[(2, 224, 224, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[(None, 56, 56, 1), (None, 56, 56, 4)],
            expected_output_dtypes=['float16'] * 2
        )


if __name__ == '__main__':
    tf.test.main()
