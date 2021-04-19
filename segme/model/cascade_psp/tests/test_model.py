import numpy as np
import tensorflow as tf
from tensorflow.python.keras import keras_parameterized, testing_utils
from tensorflow.python.training.tracking import util as trackable_util
from tensorflow.python.util import object_identity
from ..model import CascadePSP, build_cascade_psp
from ....testing_utils import layer_multi_io_test


@keras_parameterized.run_all_keras_modes
class TestCascadePSP(keras_parameterized.TestCase):
    def setUp(self):
        super(TestCascadePSP, self).setUp()
        self.default_policy = tf.keras.mixed_precision.experimental.global_policy()

    def tearDown(self):
        super(TestCascadePSP, self).tearDown()
        tf.keras.mixed_precision.experimental.set_policy(self.default_policy)

    def test_layer(self):
        # layer_multi_io_test(
        #     CascadePSP,
        #     kwargs={'psp_sizes': (1, 2, 3, 6)},
        #     input_shapes=[(2, 224, 224, 3), (2, 224, 224, 1), (2, 224, 224, 1)],
        #     input_dtypes=['uint8'] * 3,
        #     expected_output_shapes=[(None, 224, 224, 1)] * 6,
        #     expected_output_dtypes=['float32'] * 6
        # )

        tf.keras.mixed_precision.experimental.set_policy('mixed_float16')
        layer_multi_io_test(
            CascadePSP,
            kwargs={'psp_sizes': (1, 2, 3, 6)},
            input_shapes=[(2, 224, 224, 3), (2, 224, 224, 1), (2, 224, 224, 1)],
            input_dtypes=['uint8'] * 3,
            expected_output_shapes=[(None, 224, 224, 1)] * 6,
            expected_output_dtypes=['float32'] * 6
        )

    # def test_model(self):
    #     num_classes = 1
    #     model = build_cascade_psp(psp_sizes=(1, 2, 3, 6))
    #     model.compile(
    #         optimizer='sgd', loss='binary_crossentropy',
    #         run_eagerly=testing_utils.should_run_eagerly())
    #     model.fit(
    #         [
    #             np.random.random((2, 224, 224, 3)).astype(np.uint8),
    #             np.random.random((2, 224, 224, 1)).astype(np.uint8),
    #             np.random.random((2, 224, 224, 1)).astype(np.uint8),
    #         ],
    #         np.random.randint(0, num_classes, (2, 224, 224)),
    #         epochs=1, batch_size=10)
    #
    #     # test config
    #     model.get_config()
    #
    #     # check whether the model variables are present
    #     # in the trackable list of objects
    #     checkpointed_objects = object_identity.ObjectIdentitySet(trackable_util.list_objects(model))
    #     for v in model.variables:
    #         self.assertIn(v, checkpointed_objects)


if __name__ == '__main__':
    tf.test.main()
