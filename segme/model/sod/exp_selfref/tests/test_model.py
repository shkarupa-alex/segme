import numpy as np
import tensorflow as tf
from keras import mixed_precision
from keras.src.testing_infra import test_combinations, test_utils
from tensorflow.python.checkpoint import checkpoint
from tensorflow.python.util import object_identity
from segme.model.sod.exp_selfref.model import ExpSelfRef
from segme.model.sod.exp_selfref.loss import exp_self_ref_losses
from segme.testing_utils import layer_multi_io_test


@test_combinations.run_all_keras_modes
class TestExpSelfRef(test_combinations.TestCase):
    def setUp(self):
        super(TestExpSelfRef, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestExpSelfRef, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        layer_multi_io_test(
            ExpSelfRef,
            kwargs={'sup_unfold': False, 'window_size': 24},
            input_shapes=[(2, 384, 384, 3)],
            input_dtypes=['uint8'] * 1,
            expected_output_shapes=[(None, 384, 384, 1)] * 5 * 2,
            expected_output_dtypes=['float32'] * 5 * 2
        )

    def test_fp16(self):
        layer_multi_io_test(
            ExpSelfRef,
            kwargs={'sup_unfold': True, 'window_size': 24},
            input_shapes=[(2, 384, 384, 3)],
            input_dtypes=['uint8'] * 1,
            expected_output_shapes=[(None, 384, 384, 1)] * 5 * 2,
            expected_output_dtypes=['float32'] * 5 * 2
        )

    def test_model(self):
        model = ExpSelfRef()
        model.compile(
            optimizer='sgd', loss=exp_self_ref_losses(5), run_eagerly=test_utils.should_run_eagerly(),
            jit_compile=False)
        model.fit(
            np.random.random((2, 384, 384, 3)).astype(np.uint8),
            [np.random.random((2, 384, 384, 1)).astype(np.float32)] * 5 * 2,
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
