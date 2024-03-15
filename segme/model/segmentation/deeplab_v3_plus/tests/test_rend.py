import numpy as np
import tensorflow as tf
from tf_keras import mixed_precision
from tf_keras.src.testing_infra import test_combinations, test_utils
from tensorflow.python.checkpoint import checkpoint
from tensorflow.python.util import object_identity
from segme.model.segmentation.deeplab_v3_plus.rend import DeepLabV3PlusWithPointRend, build_deeplab_v3_plus_with_point_rend
from segme.testing_utils import layer_multi_io_test


@test_combinations.run_all_keras_modes
class TestDeepLabV3PlusWithPointRend(test_combinations.TestCase):
    def setUp(self):
        super(TestDeepLabV3PlusWithPointRend, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestDeepLabV3PlusWithPointRend, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        layer_multi_io_test(
            DeepLabV3PlusWithPointRend,
            kwargs={
                'classes': 4, 'aspp_filters': 8, 'aspp_stride': 32, 'low_filters': 16, 'decoder_filters': 4,
                'rend_strides': [2], 'rend_units': [4], 'rend_points': [0.1697, 0.0005], 'rend_oversample': 3,
                'rend_importance': 0.75},
            input_shapes=[(2, 224, 224, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[(None, 224, 224, 4), (None, None, 4), (None, None, 2)],
            expected_output_dtypes=['float32', 'float32', 'float32']
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        layer_multi_io_test(
            DeepLabV3PlusWithPointRend,
            kwargs={
                'classes': 1, 'aspp_filters': 8, 'aspp_stride': 32, 'low_filters': 16, 'decoder_filters': 4,
                'rend_strides': [2, 4], 'rend_units': [2, 2], 'rend_points': [0.1697, 0.0005], 'rend_oversample': 3,
                'rend_importance': 0.75},
            input_shapes=[(2, 224, 224, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[(None, 224, 224, 1), (None, None, 1), (None, None, 2)],
            expected_output_dtypes=['float32', 'float16', 'float16']
        )

    def test_model(self):
        num_classes = 5
        model = build_deeplab_v3_plus_with_point_rend(
            classes=num_classes,
            aspp_filters=8,
            aspp_stride=16,
            low_filters=16,
            decoder_filters=4,
            rend_strides=(2, 4),
            rend_units=(256,),
            rend_points=(0.1697, 0.0005),
            rend_oversample=3,
            rend_importance=0.75,
            rend_weights=True
        )
        model.compile(
            optimizer='sgd', loss=['sparse_categorical_crossentropy', None],
            run_eagerly=test_utils.should_run_eagerly())
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
        checkpointed_objects = object_identity.ObjectIdentitySet(checkpoint.list_objects(model))
        for v in model.variables:
            self.assertIn(v, checkpointed_objects)


if __name__ == '__main__':
    tf.test.main()
