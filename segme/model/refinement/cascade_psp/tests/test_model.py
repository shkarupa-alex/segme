import numpy as np
from keras.src import mixed_precision
from keras.src import testing
from tensorflow.python.checkpoint import checkpoint
from tensorflow.python.util import object_identity

from segme.model.refinement.cascade_psp.loss import cascade_psp_losses
from segme.model.refinement.cascade_psp.model import CascadePSP
from segme.model.refinement.cascade_psp.model import build_cascade_psp


class TestCascadePSP(testing.TestCase):
    def setUp(self):
        super(TestCascadePSP, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestCascadePSP, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        self.run_layer_test(
            CascadePSP,
            init_kwargs={},
            input_shape=((2, 224, 224, 3), (2, 224, 224, 1), (2, 224, 224, 1)),
            input_dtype=["uint8"] * 3,
            expected_output_shape=((None, 224, 224, 1)) * 6,
            expected_output_dtype=["float32"] * 6,
        )

    def test_fp16(self):
        mixed_precision.set_global_policy("mixed_float16")
        self.run_layer_test(
            CascadePSP,
            init_kwargs={},
            input_shape=((2, 224, 224, 3), (2, 224, 224, 1), (2, 224, 224, 1)),
            input_dtype=["uint8"] * 3,
            expected_output_shape=((None, 224, 224, 1)) * 6,
            expected_output_dtype=["float32"] * 6,
        )

    def test_model(self):
        num_classes = 1
        model = build_cascade_psp()
        model.compile(
            optimizer="sgd",
            loss=cascade_psp_losses(),
        )
        model.fit(
            [
                np.random.random((2, 224, 224, 3)).astype(np.uint8),
                np.random.random((2, 224, 224, 1)).astype(np.uint8),
                np.random.random((2, 224, 224, 1)).astype(np.uint8),
            ],
            np.random.randint(0, num_classes, (2, 224, 224, 1)),
            epochs=1,
            batch_size=10,
        )

        # test config
        model.get_config()

        # check whether the model variables are present
        # in the trackable list of objects
        checkpointed_objects = object_identity.ObjectIdentitySet(
            checkpoint.list_objects(model)
        )
        for v in model.variables:
            self.assertIn(v, checkpointed_objects)
