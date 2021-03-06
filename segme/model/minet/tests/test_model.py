import numpy as np
import tensorflow as tf
from tensorflow.python.keras import keras_parameterized, testing_utils
from tensorflow.python.training.tracking import util as trackable_util
from tensorflow.python.util import object_identity
from ..model import build_minet, MINet


@keras_parameterized.run_all_keras_modes
class TestMINet(keras_parameterized.TestCase):
    def setUp(self):
        super(TestMINet, self).setUp()
        self.default_policy = tf.keras.mixed_precision.experimental.global_policy()

    def tearDown(self):
        super(TestMINet, self).tearDown()
        tf.keras.mixed_precision.experimental.set_policy(self.default_policy)

    def test_layer(self):
        testing_utils.layer_test(
            MINet,
            kwargs={'classes': 3, 'bone_arch': 'resnet_50', 'bone_init': 'imagenet', 'bone_train': False},
            input_shape=[2, 62, 62, 3],
            input_dtype='uint8',
            expected_output_shape=[None, 62, 62, 3],
            expected_output_dtype='float32'
        )

        glob_policy = tf.keras.mixed_precision.experimental.global_policy()
        tf.keras.mixed_precision.experimental.set_policy('mixed_float16')
        testing_utils.layer_test(
            MINet,
            kwargs={'classes': 1, 'bone_arch': 'resnet_50', 'bone_init': 'imagenet', 'bone_train': False},
            input_shape=[2, 64, 64, 3],
            input_dtype='uint8',
            expected_output_shape=[None, 64, 64, 1],
            expected_output_dtype='float32'
        )
        tf.keras.mixed_precision.experimental.set_policy(glob_policy)

    def test_model(self):
        num_classes = 1
        model = build_minet(
            classes=num_classes,
            bone_arch='resnet_50',
            bone_init='imagenet',
            bone_train=False
        )
        model.compile(
            optimizer='sgd', loss='binary_crossentropy',
            run_eagerly=testing_utils.should_run_eagerly())
        model.fit(
            np.random.random((2, 224, 224, 3)).astype(np.uint8),
            np.random.randint(0, num_classes, (2, 224, 224)),
            epochs=1, batch_size=1)

        # test config
        model.get_config()

        # check whether the model variables are present
        # in the trackable list of objects
        checkpointed_objects = object_identity.ObjectIdentitySet(trackable_util.list_objects(model))
        for v in model.variables:
            self.assertIn(v, checkpointed_objects)


if __name__ == '__main__':
    tf.test.main()
