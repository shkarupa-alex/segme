import numpy as np
import tensorflow as tf
from keras.testing_infra import test_combinations, test_utils
from keras.mixed_precision import policy as mixed_precision
from tensorflow.python.training.tracking import util as trackable_util
from tensorflow.python.util import object_identity
from ..model import MatteFormer, build_matte_former
from ....testing_utils import layer_multi_io_test


@test_combinations.run_all_keras_modes
class TestMatteFormer(test_combinations.TestCase):
    def setUp(self):
        super(TestMatteFormer, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestMatteFormer, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        layer_multi_io_test(
            MatteFormer,
            kwargs={'filters': (256, 128, 64, 32), 'depths': (2, 3, 3, 2)},
            input_shapes=[(2, 512, 512, 3), (2, 512, 512, 1)],
            input_dtypes=['uint8'] * 2,
            expected_output_shapes=[(None, 512, 512, 1)] * 4,
            expected_output_dtypes=['float32'] * 4
        )

        mixed_precision.set_global_policy('mixed_float16')
        layer_multi_io_test(
            MatteFormer,
            kwargs={'filters': (256, 128, 64, 32), 'depths': (2, 3, 3, 2)},
            input_shapes=[(2, 256, 256, 3), (2, 256, 256, 1)],
            input_dtypes=['uint8'] * 2,
            expected_output_shapes=[(None, 256, 256, 1)] * 4,
            expected_output_dtypes=['float32'] * 4
        )

    def test_model(self):
        model = build_matte_former()
        model.compile(
            optimizer='sgd', loss='mse',
            run_eagerly=test_utils.should_run_eagerly())
        model.fit(
            [
                np.random.random((2, 256, 256, 3)).astype(np.uint8),
                np.random.random((2, 256, 256, 1)).astype(np.uint8)
            ],
            [
                np.random.random((2, 256, 256, 1)).astype(np.float32),
                np.random.random((2, 256, 256, 1)).astype(np.float32),
                np.random.random((2, 256, 256, 1)).astype(np.float32),
                np.random.random((2, 256, 256, 1)).astype(np.float32)
            ],
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
