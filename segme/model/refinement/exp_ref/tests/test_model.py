import numpy as np
import tensorflow as tf
from tf_keras import mixed_precision
from tf_keras.src.testing_infra import test_combinations, test_utils
from tensorflow.python.checkpoint import checkpoint
from tensorflow.python.util import object_identity
from segme.model.refinement.exp_ref.model import ExpRef
from segme.testing_utils import layer_multi_io_test


@test_combinations.run_all_keras_modes
class TestExpRef(test_combinations.TestCase):
    def setUp(self):
        super(TestExpRef, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestExpRef, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        layer_multi_io_test(
            ExpRef,
            kwargs={'sup_unfold': False},
            input_shapes=[(2, 240, 240, 3), (2, 240, 240, 1)],
            input_dtypes=['uint8'] * 2,
            expected_output_shapes=[(None, 240, 240, 1)] * 3,
            expected_output_dtypes=['float32'] * 3
        )
        layer_multi_io_test(
            ExpRef,
            kwargs={'sup_unfold': True},
            input_shapes=[(2, 240, 240, 3), (2, 240, 240, 1)],
            input_dtypes=['uint8'] * 2,
            expected_output_shapes=[(None, 240, 240, 1)] * 3,
            expected_output_dtypes=['float32'] * 3
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        layer_multi_io_test(
            ExpRef,
            kwargs={'sup_unfold': False},
            input_shapes=[(2, 240, 240, 3), (2, 240, 240, 1)],
            input_dtypes=['uint8'] * 2,
            expected_output_shapes=[(None, 240, 240, 1)] * 3,
            expected_output_dtypes=['float32'] * 3
        )
        layer_multi_io_test(
            ExpRef,
            kwargs={'sup_unfold': True},
            input_shapes=[(2, 240, 240, 3), (2, 240, 240, 1)],
            input_dtypes=['uint8'] * 2,
            expected_output_shapes=[(None, 240, 240, 1)] * 3,
            expected_output_dtypes=['float32'] * 3
        )

    def test_model(self):
        model = ExpRef()
        model.compile(optimizer='sgd', loss='mse', run_eagerly=test_utils.should_run_eagerly(), jit_compile=False)
        model.fit(
            [np.random.random((2, 240, 240, 3)).astype(np.uint8), np.random.random((2, 240, 240, 1)).astype(np.uint8)],
            np.random.random((2, 240, 240, 1)).astype(np.float32),
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
