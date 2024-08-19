from keras.src import models
from keras.src import testing

from segme.common.backbone import Backbone
from segme.policy import bbpol


class TestBackbone(testing.TestCase):
    def setUp(self):
        super(TestBackbone, self).setUp()
        self.default_bb = bbpol.global_policy()

    def tearDown(self):
        super(TestBackbone, self).tearDown()
        bbpol.set_global_policy(self.default_bb)

    def test_layer(self):
        self.run_layer_test(
            Backbone,
            init_kwargs={"scales": None},
            input_shape=(2, 224, 224, 3),
            input_dtype="float32",
            expected_output_shape=(
                (2, 112, 112, 64),
                (2, 56, 56, 256),
                (2, 28, 28, 512),
                (2, 14, 14, 1024),
                (2, 7, 7, 2048),
            ),
            expected_output_dtype=("float32",) * 5,
        )
        self.run_layer_test(
            Backbone,
            init_kwargs={"scales": [2, 8]},
            input_shape=(2, 224, 224, 3),
            input_dtype="uint8",
            expected_output_shape=((2, 112, 112, 64), (2, 28, 28, 512)),
            expected_output_dtype=("float32",) * 2,
        )

        with bbpol.policy_scope("swin_tiny_224-none"):
            self.run_layer_test(
                Backbone,
                init_kwargs={"scales": None},
                input_shape=(2, 224, 224, 3),
                input_dtype="float32",
                expected_output_shape=(
                    (2, 56, 56, 96),
                    (2, 28, 28, 192),
                    (2, 14, 14, 384),
                    (2, 7, 7, 768),
                ),
                expected_output_dtype=("float32",) * 4,
            )
            self.run_layer_test(
                Backbone,
                init_kwargs={"scales": None},
                input_shape=(2, 224, 224, 3),
                input_dtype="uint8",
                expected_output_shape=(
                    (2, 56, 56, 96),
                    (2, 28, 28, 192),
                    (2, 14, 14, 384),
                    (2, 7, 7, 768),
                ),
                expected_output_dtype=("float32",) * 4,
            )

    def test_policy_scope_memorize(self):
        with bbpol.policy_scope("swin_tiny_224-none"):
            boneinst = Backbone()
        boneinst.build([None, None, None, 3])

        restored = models.Model.from_config(boneinst.get_config())
        restored.build([None, None, None, 3])

        self.assertEqual(
            len(restored.trainable_weights), len(boneinst.trainable_weights)
        )
        self.assertEqual(
            len(restored.non_trainable_weights),
            len(boneinst.non_trainable_weights),
        )
