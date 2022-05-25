import numpy as np
import tensorflow as tf
from keras.testing_infra import test_combinations, test_utils
from keras.mixed_precision import policy as mixed_precision
from tensorflow.python.training.tracking import util as trackable_util
from tensorflow.python.util import object_identity
from ..model import build_dexi_ned, DexiNed
from ....testing_utils import layer_multi_io_test


@test_combinations.run_all_keras_modes
class TestDexiNed(test_combinations.TestCase):
    def setUp(self):
        super(TestDexiNed, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestDexiNed, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        layer_multi_io_test(
            DexiNed,
            kwargs={'classes': 1},
            input_shapes=[(2, 224, 224, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[(None, 224, 224, 1)] * 7,
            expected_output_dtypes=['float32'] * 7
        )
        layer_multi_io_test(
            DexiNed,
            kwargs={'classes': 3},
            input_shapes=[(2, 224, 224, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[(None, 224, 224, 3)] * 7,
            expected_output_dtypes=['float32'] * 7
        )

        mixed_precision.set_global_policy('mixed_float16')
        layer_multi_io_test(
            DexiNed,
            kwargs={'classes': 3},
            input_shapes=[(2, 224, 224, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[(None, 224, 224, 3)] * 7,
            expected_output_dtypes=['float32'] * 7
        )

    def test_model(self):
        num_classes = 1
        model = build_dexi_ned(num_classes)
        model.compile(
            optimizer='sgd', loss='binary_crossentropy',
            run_eagerly=test_utils.should_run_eagerly())
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
