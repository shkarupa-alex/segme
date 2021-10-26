import numpy as np
import tensorflow as tf
from keras import keras_parameterized, testing_utils
from keras.mixed_precision import policy as mixed_precision
from tensorflow.python.training.tracking import util as trackable_util
from tensorflow.python.util import object_identity
from ..model import build_tri_trans_net, TriTransNet
from ....testing_utils import layer_multi_io_test


@keras_parameterized.run_all_keras_modes
class TestTriTransNet(keras_parameterized.TestCase):
    def setUp(self):
        super(TestTriTransNet, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestTriTransNet, self).tearDown()
        mixed_precision.set_policy(self.default_policy)

    def test_layer(self):
        layer_multi_io_test(
            TriTransNet,
            kwargs={},
            input_shapes=[(2, 256, 256, 3), (2, 256, 256, 1)],
            input_dtypes=['uint8', 'uint16'],
            expected_output_shapes=[(None, 256, 256, 1)] * 4,
            expected_output_dtypes=['float32'] * 4
        )

        # TODO: Too slow
        # mixed_precision.set_policy('mixed_float16')
        # layer_multi_io_test(
        #     TriTransNet,
        #     kwargs={},
        #     input_shapes=[(2, 128, 128, 3), (2, 128, 128, 1)],
        #     input_dtypes=['uint8', 'uint16'],
        #     expected_output_shapes=[(None, 128, 128, 1)] * 4,
        #     expected_output_dtypes=['float32'] * 4
        # )

    def test_model(self):
        model = build_tri_trans_net(image_size=128)
        model.compile(
            optimizer='sgd', loss='binary_crossentropy',
            run_eagerly=testing_utils.should_run_eagerly())
        model.fit(
            [np.random.random((2, 128, 128, 3)).astype(np.uint8), np.random.random((2, 128, 128, 1)).astype(np.uint16)],
            np.random.randint(0, 1, (2, 128, 128)),
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
