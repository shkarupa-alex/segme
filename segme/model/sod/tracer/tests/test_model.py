import numpy as np
import tensorflow as tf
from keras import mixed_precision
from keras.src.testing_infra import test_combinations, test_utils
from tensorflow.python.checkpoint import checkpoint
from tensorflow.python.util import object_identity
from segme.policy import cnapol
from segme.model.sod.tracer.model import build_tracer, Tracer
from segme.model.sod.tracer.loss import tracer_losses
from segme.testing_utils import layer_multi_io_test


@test_combinations.run_all_keras_modes
class TestTracer(test_combinations.TestCase):
    def setUp(self):
        super(TestTracer, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestTracer, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        layer_multi_io_test(
            Tracer,
            kwargs={'radius': 16, 'confidence': 0.1, 'rfb': (32, 64, 128), 'denoise': 0.93},
            input_shapes=[(2, 64, 64, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[(None, 64, 64, 1)] * 5,
            expected_output_dtypes=['float32'] * 5
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        layer_multi_io_test(
            Tracer,
            kwargs={'radius': 16, 'confidence': 0.1, 'rfb': (32, 64, 128), 'denoise': 0.93},
            input_shapes=[(2, 64, 64, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[(None, 64, 64, 1)] * 5,
            expected_output_dtypes=['float32'] * 5
        )

    def test_model(self):
        with cnapol.policy_scope('conv-bn-selu'):
            model = build_tracer()
            model.compile(
                optimizer='sgd', loss=tracer_losses(),
                run_eagerly=test_utils.should_run_eagerly())
            model.fit(
                np.random.random((2, 224, 224, 3)).astype(np.uint8),
                [np.random.randint(0, 1, (2, 224, 224, 1)) for _ in range(5)],
                epochs=1, batch_size=1)

            # test config
            model.get_config()

            # check whether the model variables are present
            # in the trackable list of objects
            checkpointed_objects = object_identity.ObjectIdentitySet(checkpoint.list_objects(model))
            for v in model.variables:
                self.assertIn(v, checkpointed_objects)


if __name__ == '__main__':
    tf.test.main()
