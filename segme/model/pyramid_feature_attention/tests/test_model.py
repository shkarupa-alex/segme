import numpy as np
import tensorflow as tf
from tensorflow.python.keras import keras_parameterized, testing_utils
from tensorflow.python.training.tracking import util as trackable_util
from tensorflow.python.util import object_identity
from ..model import build_pyramid_feature_attention, PyramidFeatureAttention


@keras_parameterized.run_all_keras_modes
class TestPyramidFeatureAttention(keras_parameterized.TestCase):
    def test_layer(self):
        testing_utils.layer_test(
            PyramidFeatureAttention,
            kwargs={'bone_arch': 'vgg_16', 'bone_init': 'imagenet', 'bone_train': False},
            input_shape=[2, 64, 64, 3],
            input_dtype='uint8',
            expected_output_shape=[None, 64, 64, 128],
            expected_output_dtype='float32'
        )

    def test_model(self):
        num_classes = 2

        model = build_pyramid_feature_attention(
            channels=3,
            classes=num_classes,
            bone_arch='vgg_16',
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
