import numpy as np
import tensorflow as tf
from keras import mixed_precision
from keras.src.testing_infra import test_combinations, test_utils
from tensorflow.python.checkpoint import checkpoint
from tensorflow.python.util import object_identity
from segme.model.sod.exp_sod.model import ExpSOD
from segme.model.sod.exp_sod.loss import exp_sod_losses
from segme.testing_utils import layer_multi_io_test


@test_combinations.run_all_keras_modes
class TestExpSOD(test_combinations.TestCase):
    def setUp(self):
        super(TestExpSOD, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestExpSOD, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        layer_multi_io_test(
            ExpSOD,
            kwargs={
                'sup_unfold': False, 'with_depth': False, 'with_unknown': False, 'transform_depth': 3,
                'window_size': 24, 'path_gamma': 0.01, 'path_drop': 0.2},
            input_shapes=[(2, 384, 384, 3)],
            input_dtypes=['uint8'] * 1,
            expected_output_shapes=[(None, 384, 384, 1)] * 5,
            expected_output_dtypes=['float32'] * 5
        )
        layer_multi_io_test(
            ExpSOD,
            kwargs={
                'sup_unfold': True, 'with_depth': False, 'with_unknown': False, 'transform_depth': 3,
                'window_size': 24, 'path_gamma': 0.01, 'path_drop': 0.2},
            input_shapes=[(2, 384, 384, 3)],
            input_dtypes=['uint8'] * 1,
            expected_output_shapes=[(None, 384, 384, 1)] * 5,
            expected_output_dtypes=['float32'] * 5
        )
        layer_multi_io_test(
            ExpSOD,
            kwargs={
                'sup_unfold': False, 'with_depth': True, 'with_unknown': False, 'transform_depth': 3,
                'window_size': 24, 'path_gamma': 0.01, 'path_drop': 0.2},
            input_shapes=[(2, 384, 384, 3)],
            input_dtypes=['uint8'] * 1,
            expected_output_shapes=[(None, 384, 384, 1)] * 5 * 2,
            expected_output_dtypes=['float32'] * 5 * 2
        )
        layer_multi_io_test(
            ExpSOD,
            kwargs={
                'sup_unfold': True, 'with_depth': False, 'with_unknown': True, 'transform_depth': 3,
                'window_size': 24, 'path_gamma': 0.01, 'path_drop': 0.2},
            input_shapes=[(2, 384, 384, 3)],
            input_dtypes=['uint8'] * 1,
            expected_output_shapes=[(None, 384, 384, 1)] * 5 * 2,
            expected_output_dtypes=['float32'] * 5 * 2
        )

    def test_fp16(self):
        layer_multi_io_test(
            ExpSOD,
            kwargs={
                'sup_unfold': True, 'with_depth': True, 'with_unknown': True, 'transform_depth': 3,
                'window_size': 24, 'path_gamma': 0.01, 'path_drop': 0.2},
            input_shapes=[(2, 384, 384, 3)],
            input_dtypes=['uint8'] * 1,
            expected_output_shapes=[(None, 384, 384, 1)] * 5 * 3,
            expected_output_dtypes=['float32'] * 5 * 3
        )

    def test_model(self):
        model = ExpSOD(with_depth=True, with_unknown=True)
        model.compile(
            optimizer='sgd', loss=exp_sod_losses(5, with_depth=True, with_unknown=True),
            run_eagerly=test_utils.should_run_eagerly(), jit_compile=False)
        model.fit(
            np.random.random((2, 384, 384, 3)).astype(np.uint8),
            [np.random.random((2, 384, 384, 1)).astype(np.float32)] * 5 * 3,
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
