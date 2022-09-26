import numpy as np
import tensorflow as tf
from keras.testing_infra import test_combinations, test_utils
from keras.mixed_precision import policy as mixed_precision
from tensorflow.python.training.tracking import util as trackable_util
from tensorflow.python.util import object_identity
from segme.model.segmentation.deeplab_v3_plus.hattn import DeepLabV3PlusWithHierarchicalAttention, \
    build_deeplab_v3_plus_with_hierarchical_attention


@test_combinations.run_all_keras_modes
class TestDeepLabV3PlusWithHierarchicalAttention(test_combinations.TestCase):
    def setUp(self):
        super(TestDeepLabV3PlusWithHierarchicalAttention, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestDeepLabV3PlusWithHierarchicalAttention, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            DeepLabV3PlusWithHierarchicalAttention,
            kwargs={
                'classes': 4, 'aspp_filters': 8, 'aspp_stride': 32, 'low_filters': 16, 'decoder_filters': 5,
                'scales': ((0.5,), (0.25, 0.5, 2.0))},
            input_shape=(2, 224, 224, 3),
            input_dtype='uint8',
            expected_output_shape=(None, 224, 224, 4),
            expected_output_dtype='float32'
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        test_utils.layer_test(
            DeepLabV3PlusWithHierarchicalAttention,
            kwargs={
                'classes': 1, 'aspp_filters': 8, 'aspp_stride': 32, 'low_filters': 16, 'decoder_filters': 4,
                'scales': ((0.5,), (0.25, 0.5, 2.0))},
            input_shape=(2, 224, 224, 3),
            input_dtype='uint8',
            expected_output_shape=(None, 224, 224, 1),
            expected_output_dtype='float32'
        )

    def test_model(self):
        num_classes = 5
        model = build_deeplab_v3_plus_with_hierarchical_attention(
            classes=num_classes,
            aspp_filters=8,
            aspp_stride=16,
            low_filters=16,
            decoder_filters=4,
            scales=((0.5,), (0.25, 0.5, 2.0)))
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
