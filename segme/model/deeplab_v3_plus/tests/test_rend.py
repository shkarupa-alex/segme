import numpy as np
import tensorflow as tf
from tensorflow.python.keras import keras_parameterized, testing_utils
from tensorflow.python.training.tracking import util as trackable_util
from tensorflow.python.util import object_identity
from ..rend import DeepLabV3PlusWithPointRend, build_deeplab_v3_plus_with_point_rend
from ....testing_utils import layer_multi_io_test


@keras_parameterized.run_all_keras_modes
class TestDeepLabV3PlusWithPointRend(keras_parameterized.TestCase):
    def setUp(self):
        super(TestDeepLabV3PlusWithPointRend, self).setUp()
        self.default_policy = tf.keras.mixed_precision.experimental.global_policy()

    def tearDown(self):
        super(TestDeepLabV3PlusWithPointRend, self).tearDown()
        tf.keras.mixed_precision.experimental.set_policy(self.default_policy)

    def test_layer(self):
        # TODO: wait for issue with Sequential model restoring
        #  will be resolved to migrate back on testing_utils.layer_test
        layer_multi_io_test(
            DeepLabV3PlusWithPointRend,
            kwargs={
                'classes': 4, 'bone_arch': 'resnet_50', 'bone_init': 'imagenet', 'bone_train': False,
                'aspp_filters': 8, 'aspp_stride': 32, 'low_filters': 16, 'decoder_filters': 4,
                'rend_strides': [2], 'rend_units': [4], 'rend_points': [0.1697, 0.0005], 'rend_oversample': 3,
                'rend_importance': 0.75, 'rend_corners': True},
            input_shapes=[(2, 224, 224, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[(None, 224, 224, 4), (None, None, 4), (None, None, 2)],
            expected_output_dtypes=['float32', 'float32', 'float32']
        )

        tf.keras.mixed_precision.experimental.set_policy('mixed_float16')
        layer_multi_io_test(
            DeepLabV3PlusWithPointRend,
            kwargs={
                'classes': 1, 'bone_arch': 'resnet_50', 'bone_init': 'imagenet', 'bone_train': False,
                'aspp_filters': 8, 'aspp_stride': 32, 'low_filters': 16, 'decoder_filters': 4,
                'rend_strides': [2, 4], 'rend_units': [2, 2], 'rend_points': [0.1697, 0.0005], 'rend_oversample': 3,
                'rend_importance': 0.75, 'rend_corners': False},
            input_shapes=[(2, 224, 224, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[(None, 224, 224, 1), (None, None, 1), (None, None, 2)],
            expected_output_dtypes=['float32', 'float16', 'float16']
        )

    def test_model(self):
        num_classes = 5
        model = build_deeplab_v3_plus_with_point_rend(
            classes=num_classes,
            bone_arch='resnet_50',
            bone_init='imagenet',
            bone_train=False,
            aspp_filters=8,
            aspp_stride=16,
            low_filters=16,
            decoder_filters=4,
            rend_strides=(2, 4),
            rend_units=(256,),
            rend_points=(0.1697, 0.0005),
            rend_oversample=3,
            rend_importance=0.75,
            rend_weights=True,
            rend_corners=True
        )
        model.compile(
            optimizer='sgd', loss=['sparse_categorical_crossentropy', None],
            run_eagerly=testing_utils.should_run_eagerly())
        model.fit(
            {
                'image': np.random.random((2, 224, 224, 3)).astype(np.uint8),
                'label': np.random.randint(0, num_classes, (2, 224, 224)),
                'weight': np.random.rand(2, 224, 224),
            },
            np.random.randint(0, num_classes, (2, 224, 224)),
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
