import numpy as np
import tensorflow as tf
from keras import keras_parameterized, testing_utils
from keras.mixed_precision import policy as mixed_precision
from tensorflow.python.training.tracking import util as trackable_util
from tensorflow.python.util import object_identity
from ..model import FBAMatting, build_fba_matting
from ....testing_utils import layer_multi_io_test


@keras_parameterized.run_all_keras_modes
class TestFBAMatting(keras_parameterized.TestCase):
    def setUp(self):
        super(TestFBAMatting, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestFBAMatting, self).tearDown()
        mixed_precision.set_policy(self.default_policy)

    def test_layer(self):
        layer_multi_io_test(
            FBAMatting,
            kwargs={'pool_scales': (1, 2, 3, 6)},
            input_shapes=[(2, 128, 128, 3), (2, 128, 128, 1)],
            input_dtypes=['uint8'] * 2,
            expected_output_shapes=[(None, 128, 128, 7), (None, 128, 128, 1), (None, 128, 128, 3), (None, 128, 128, 3)],
            expected_output_dtypes=['float32'] * 4
        )

        mixed_precision.set_policy('mixed_float16')
        layer_multi_io_test(
            FBAMatting,
            kwargs={'pool_scales': (1, 2, 3, 6)},
            input_shapes=[(2, 128, 128, 3), (2, 128, 128, 1)],
            input_dtypes=['uint8'] * 2,
            expected_output_shapes=[(None, 128, 128, 7), (None, 128, 128, 1), (None, 128, 128, 3), (None, 128, 128, 3)],
            expected_output_dtypes=['float32', 'float16', 'float16', 'float16']
        )

    def test_model(self):
        model = build_fba_matting(psp_sizes=(1, 2, 3, 6))
        model.compile(
            optimizer='sgd', loss=['mse', None, None, None],
            run_eagerly=testing_utils.should_run_eagerly())
        model.fit(
            [
                np.random.random((2, 224, 224, 3)).astype(np.uint8),
                np.random.random((2, 224, 224, 1)).astype(np.uint8),
            ],
            [
                np.random.random((2, 224, 224, 7)).astype(np.float32),
                np.random.random((2, 224, 224, 1)).astype(np.float32),
                np.random.random((2, 224, 224, 3)).astype(np.float32),
                np.random.random((2, 224, 224, 3)).astype(np.float32)
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
