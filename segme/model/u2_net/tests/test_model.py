import numpy as np
import tensorflow as tf
from tensorflow.python.keras import keras_parameterized, testing_utils
from tensorflow.python.training.tracking import util as trackable_util
from tensorflow.python.util import object_identity
from ..model import U2Net, U2NetP, build_u2_net, build_u2_netp
from ....testing_utils import layer_multi_io_test


@keras_parameterized.run_all_keras_modes
class TestU2Net(keras_parameterized.TestCase):
    def setUp(self):
        super(TestU2Net, self).setUp()
        self.default_policy = tf.keras.mixed_precision.experimental.global_policy()

    def tearDown(self):
        super(TestU2Net, self).tearDown()
        tf.keras.mixed_precision.experimental.set_policy(self.default_policy)

    def test_u2net(self):
        layer_multi_io_test(
            U2Net,
            kwargs={'classes': 1},
            input_shapes=[(2, 64, 64, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[(None, 64, 64, 1)] * 7,
            expected_output_dtypes=['float32'] * 7
        )

        glob_policy = tf.keras.mixed_precision.experimental.global_policy()
        tf.keras.mixed_precision.experimental.set_policy('mixed_float16')
        layer_multi_io_test(
            U2Net,
            kwargs={'classes': 2},
            input_shapes=[(2, 64, 64, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[(None, 64, 64, 2)] * 7,
            expected_output_dtypes=['float32'] * 7
        )
        tf.keras.mixed_precision.experimental.set_policy(glob_policy)

    def test_u2netp(self):
        layer_multi_io_test(
            U2NetP,
            kwargs={'classes': 3},
            input_shapes=[(2, 64, 64, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[(None, 64, 64, 3)] * 7,
            expected_output_dtypes=['float32'] * 7
        )

        glob_policy = tf.keras.mixed_precision.experimental.global_policy()
        tf.keras.mixed_precision.experimental.set_policy('mixed_float16')
        layer_multi_io_test(
            U2NetP,
            kwargs={'classes': 3},
            input_shapes=[(2, 64, 64, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[(None, 64, 64, 3)] * 7,
            expected_output_dtypes=['float32'] * 7
        )
        tf.keras.mixed_precision.experimental.set_policy(glob_policy)

    def test_model(self):
        num_classes = 2
        model = build_u2_net(channels=3, classes=num_classes)
        model.compile(
            optimizer='sgd', loss='sparse_categorical_crossentropy',
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

    def test_model_p(self):
        num_classes = 1
        model = build_u2_netp(channels=3, classes=num_classes)
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
