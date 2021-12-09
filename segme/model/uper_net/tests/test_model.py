import numpy as np
import tensorflow as tf
from keras import keras_parameterized, testing_utils
from keras.mixed_precision import policy as mixed_precision
from tensorflow.python.training.tracking import util as trackable_util
from tensorflow.python.util import object_identity
from ..model import UPerNet, build_uper_net
from ....testing_utils import layer_multi_io_test


@keras_parameterized.run_all_keras_modes
class TestUPerNet(keras_parameterized.TestCase):
    def setUp(self):
        super(TestUPerNet, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestUPerNet, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        testing_utils.layer_test(
            UPerNet,
            kwargs={'classes': 1, 'bone_arch': 'swin_tiny_224', 'bone_init': 'imagenet', 'bone_train': False,
                    'dropout': 0., 'dec_filters': 8, 'psp_sizes': (1, 2, 3, 6), 'aux_filters': 4},
            input_shape=(2, 240, 240, 3),
            input_dtype='uint8',
            expected_output_shape=(None, 240, 240, 1),
            expected_output_dtype='float32'
        )

        mixed_precision.set_global_policy('mixed_float16')
        layer_multi_io_test(
            UPerNet,
            kwargs={'classes': 4, 'bone_arch': 'swin_tiny_224', 'bone_init': 'imagenet', 'bone_train': True,
                    'dropout': 0.1, 'dec_filters': 8, 'psp_sizes': (1, 2, 3, 6), 'aux_filters': 4},
            input_shapes=[(2, 240, 240, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[(None, 240, 240, 4), (None, 240, 240, 4)],
            expected_output_dtypes=['float32'] * 2
        )

    def test_model(self):
        model = build_uper_net(classes=2, bone_arch='swin_tiny_224', bone_init='imagenet', bone_train=False)
        model.compile(optimizer='sgd', loss='mse', run_eagerly=testing_utils.should_run_eagerly())
        model.fit(
            np.random.random((2, 240, 240, 3)).astype(np.uint8),
            np.random.random((2, 240, 240, 2)).astype(np.float32),
            epochs=1, batch_size=10)

        # test config
        model.get_config()

        # check whether the model variables are present
        # in the trackable list of objects
        checkpointed_objects = object_identity.ObjectIdentitySet(trackable_util.list_objects(model))
        for v in model.variables:
            self.assertIn(v, checkpointed_objects)


if __name__ == '__main__':
    tf.test.main()
