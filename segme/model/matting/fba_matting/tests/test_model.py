import numpy as np
import tensorflow as tf
from keras import mixed_precision
from keras.src.testing_infra import test_combinations, test_utils
from tensorflow.python.training.tracking import util as trackable_util
from tensorflow.python.util import object_identity
from segme.policy import cnapol
from segme.model.matting.fba_matting.model import FBAMatting
from segme.model.matting.fba_matting.loss import fba_matting_losses
from segme.testing_utils import layer_multi_io_test


@test_combinations.run_all_keras_modes
class TestFBAMatting(test_combinations.TestCase):
    def setUp(self):
        super(TestFBAMatting, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestFBAMatting, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        layer_multi_io_test(
            FBAMatting,
            kwargs={},
            input_shapes=[(2, 120, 120, 3), (2, 120, 120, 2), (2, 120, 120, 6)],
            input_dtypes=['uint8'] * 3,
            expected_output_shapes=[(None, 120, 120, 7), (None, 120, 120, 1), (None, 120, 120, 3), (None, 120, 120, 3)],
            expected_output_dtypes=['float32'] * 4
        )

    # def test_fp16(self):
    #     mixed_precision.set_global_policy('mixed_float16')
    #     layer_multi_io_test(
    #         FBAMatting,
    #         kwargs={},
    #         input_shapes=[(2, 120, 120, 3), (2, 120, 120, 2), (2, 120, 120, 6)],
    #         input_dtypes=['uint8'] * 3,
    #         expected_output_shapes=[(None, 120, 120, 7), (None, 120, 120, 1), (None, 120, 120, 3), (None, 120, 120, 3)],
    #         expected_output_dtypes=['float32', 'float32', 'float32', 'float32']
    #     )

    # def test_model(self):
    #     with cnapol.policy_scope('stdconv-gn-leakyrelu'):
    #         model = build_fba_matting()
    #         model.compile(
    #             optimizer='sgd', loss=fba_matting_losses(),
    #             run_eagerly=test_utils.should_run_eagerly())
    #         model.fit(
    #             [
    #                 np.random.random((2, 240, 240, 3)).astype(np.uint8),
    #                 np.random.random((2, 240, 240, 2)).astype(np.uint8),
    #                 np.random.random((2, 240, 240, 6)).astype(np.uint8),
    #             ],
    #             [
    #                 np.random.random((2, 240, 240, 7)).astype(np.float32),
    #                 np.random.random((2, 240, 240, 1)).astype(np.float32),
    #                 np.random.random((2, 240, 240, 3)).astype(np.float32),
    #                 np.random.random((2, 240, 240, 3)).astype(np.float32)
    #             ],
    #             epochs=1, batch_size=10)
    #
    #         # test config
    #         model.get_config()
    #
    #         # check whether the model variables are present
    #         # in the trackable list of objects
    #         checkpointed_objects = object_identity.ObjectIdentitySet(trackable_util.list_objects(model))
    #         for v in model.variables:
    #             self.assertIn(v, checkpointed_objects)


if __name__ == '__main__':
    tf.test.main()
