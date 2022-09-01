import numpy as np
import tensorflow as tf
from keras.testing_infra import test_combinations, test_utils
from keras.mixed_precision import policy as mixed_precision
from tensorflow.python.training.tracking import util as trackable_util
from tensorflow.python.util import object_identity
from segme.model.cascade_psp.model import CascadePSP, build_cascade_psp
from segme.model.cascade_psp.loss import cascade_psp_losses
from segme.testing_utils import layer_multi_io_test


@test_combinations.run_all_keras_modes
class TestCascadePSP(test_combinations.TestCase):
    def setUp(self):
        super(TestCascadePSP, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestCascadePSP, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        layer_multi_io_test(
            CascadePSP,
            kwargs={},
            input_shapes=[(2, 224, 224, 3), (2, 224, 224, 1), (2, 224, 224, 1)],
            input_dtypes=['uint8'] * 3,
            expected_output_shapes=[(None, 224, 224, 1)] * 6,
            expected_output_dtypes=['float32'] * 6
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        layer_multi_io_test(
            CascadePSP,
            kwargs={},
            input_shapes=[(2, 224, 224, 3), (2, 224, 224, 1), (2, 224, 224, 1)],
            input_dtypes=['uint8'] * 3,
            expected_output_shapes=[(None, 224, 224, 1)] * 6,
            expected_output_dtypes=['float32'] * 6
        )

    def test_model(self):
        num_classes = 1
        model = build_cascade_psp()
        model.compile(
            optimizer='sgd', loss=cascade_psp_losses(),
            run_eagerly=test_utils.should_run_eagerly())
        model.fit(
            [
                np.random.random((2, 224, 224, 3)).astype(np.uint8),
                np.random.random((2, 224, 224, 1)).astype(np.uint8),
                np.random.random((2, 224, 224, 1)).astype(np.uint8),
            ],
            np.random.randint(0, num_classes, (2, 224, 224, 1)),
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
