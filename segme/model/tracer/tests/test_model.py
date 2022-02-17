import numpy as np
import tensorflow as tf
from keras import keras_parameterized, testing_utils
from keras.mixed_precision import policy as mixed_precision
from tensorflow.python.training.tracking import util as trackable_util
from tensorflow.python.util import object_identity
from ..model import build_tracer, Tracer
from ....testing_utils import layer_multi_io_test


@keras_parameterized.run_all_keras_modes
class TestTracer(keras_parameterized.TestCase):
    def setUp(self):
        super(TestTracer, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestTracer, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        layer_multi_io_test(
            Tracer,
            kwargs={'bone_arch': 'resnet_50', 'bone_init': 'imagenet', 'bone_train': False, 'radius': 16,
                    'confidence': 0.1, 'rfb': (32, 64, 128), 'denoise': 0.93},
            input_shapes=[(2, 64, 64, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[(None, 64, 64, 1)] * 5,
            expected_output_dtypes=['float32'] * 5
        )

        mixed_precision.set_global_policy('mixed_float16')
        layer_multi_io_test(
            Tracer,
            kwargs={'bone_arch': 'resnet_50', 'bone_init': 'imagenet', 'bone_train': False, 'radius': 16,
                    'confidence': 0.1, 'rfb': (32, 64, 128), 'denoise': 0.93},
            input_shapes=[(2, 64, 64, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[(None, 64, 64, 1)] * 5,
            expected_output_dtypes=['float32'] * 5
        )

    def test_model(self):
        num_classes = 1
        model = build_tracer(
            bone_arch='resnet_50',
            bone_init='imagenet',
            bone_train=False
        )
        model.compile(
            optimizer='sgd', loss='binary_crossentropy',
            run_eagerly=testing_utils.should_run_eagerly())
        model.fit(
            np.random.random((2, 224, 224, 3)).astype(np.uint8),
            [np.random.randint(0, num_classes, (2, 224, 224)) for _ in range(5)],
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
