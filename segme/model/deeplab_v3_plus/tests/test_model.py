import numpy as np
import tensorflow as tf
from keras.testing_infra import test_combinations, test_utils
from keras.mixed_precision import policy as mixed_precision
from tensorflow.python.training.tracking import util as trackable_util
from tensorflow.python.util import object_identity
from ..model import DeepLabV3Plus, build_deeplab_v3_plus
from ....testing_utils import layer_multi_io_test


@test_combinations.run_all_keras_modes
class TestDeepLabV3Plus(test_combinations.TestCase):
    def setUp(self):
        super(TestDeepLabV3Plus, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestDeepLabV3Plus, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        # TODO: wait for issue with Sequential model restoring
        #  will be resolved to migrate back on test_utils.layer_test
        layer_multi_io_test(
            DeepLabV3Plus,
            kwargs={
                'classes': 4, 'bone_arch': 'resnet_50', 'bone_init': 'imagenet', 'bone_train': False,
                'aspp_filters': 8, 'aspp_stride': 32, 'low_filters': 16, 'decoder_filters': 4},
            input_shapes=[(2, 224, 224, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[(None, 224, 224, 4)],
            expected_output_dtypes=['float32']
        )

        mixed_precision.set_global_policy('mixed_float16')
        layer_multi_io_test(
            DeepLabV3Plus,
            kwargs={
                'classes': 1, 'bone_arch': 'resnet_50', 'bone_init': 'imagenet', 'bone_train': False,
                'aspp_filters': 8, 'aspp_stride': 32, 'low_filters': 16, 'decoder_filters': 4},
            input_shapes=[(2, 224, 224, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[(None, 224, 224, 1)],
            expected_output_dtypes=['float32']
        )

    def test_model(self):
        num_classes = 5
        model = build_deeplab_v3_plus(
            classes=num_classes,
            bone_arch='resnet_50',
            bone_init='imagenet',
            bone_train=False,
            aspp_filters=8,
            aspp_stride=16,
            low_filters=16,
            decoder_filters=4
        )
        model.compile(
            optimizer='sgd', loss='sparse_categorical_crossentropy',
            run_eagerly=test_utils.should_run_eagerly())
        model.fit(
            np.random.random((2, 224, 224, 3)).astype(np.uint8),
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
