import numpy as np
import tensorflow as tf
from absl.testing import parameterized
from keras import mixed_precision
from keras.src.testing_infra import test_combinations, test_utils
from tensorflow.python.util import object_identity
from tensorflow.python.checkpoint import checkpoint
from segme.common.backbone import Backbone
from segme.policy.backbone.diy.hardswin import HardSwinTiny
from segme.policy import cnapol
from segme.testing_utils import layer_multi_io_test


@test_combinations.run_all_keras_modes
class TestModel(test_combinations.TestCase):
    def setUp(self):
        super(TestModel, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestModel, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    @staticmethod
    def _values_from_config(cfg, cls, prop):
        if isinstance(cls, str):
            cls = [cls]

        values = []
        for layer in cfg['layers']:
            if layer['registered_name'] not in cls:
                continue

            if prop not in layer['config']:
                continue

            values.append((layer['name'], layer['config'][prop]))

        return values

    def test_drop_path(self):
        config = HardSwinTiny(
            embed_dim=64, stage_depths=(2, 2, 6, 4), pretrain_window=12, weights=None, include_top=False,
            input_shape=(None, None, 3)).get_config()

        expected_drops = [
            ('stage_0_attn_0_swin_drop', 0.0), ('stage_0_attn_0_mlp_drop', 0.0),
            ('stage_0_attn_1_swin_drop', 0.015384615384615385), ('stage_0_attn_1_mlp_drop', 0.015384615384615385),

            ('stage_1_attn_0_swin_drop', 0.03076923076923077), ('stage_1_attn_0_mlp_drop', 0.03076923076923077),
            ('stage_1_attn_1_swin_drop', 0.046153846153846156), ('stage_1_attn_1_mlp_drop', 0.046153846153846156),

            ('stage_2_attn_0_swin_drop', 0.06153846153846154), ('stage_2_attn_0_mlp_drop', 0.06153846153846154),
            ('stage_2_attn_1_swin_drop', 0.07692307692307693), ('stage_2_attn_1_mlp_drop', 0.07692307692307693),
            ('stage_2_attn_2_swin_drop', 0.09230769230769231), ('stage_2_attn_2_mlp_drop', 0.09230769230769231),
            ('stage_2_attn_3_swin_drop', 0.1076923076923077), ('stage_2_attn_3_mlp_drop', 0.1076923076923077),
            ('stage_2_attn_4_swin_drop', 0.12307692307692308), ('stage_2_attn_4_mlp_drop', 0.12307692307692308),
            ('stage_2_attn_5_swin_drop', 0.13846153846153847), ('stage_2_attn_5_mlp_drop', 0.13846153846153847),

            ('stage_3_attn_0_swin_drop', 0.15384615384615385), ('stage_3_attn_0_mlp_drop', 0.15384615384615385),
            ('stage_3_attn_1_swin_drop', 0.16923076923076924), ('stage_3_attn_1_mlp_drop', 0.16923076923076924),
            ('stage_3_attn_2_swin_drop', 0.18461538461538463), ('stage_3_attn_2_mlp_drop', 0.18461538461538463),
            ('stage_3_attn_3_swin_drop', 0.2), ('stage_3_attn_3_mlp_drop', 0.2)]

        actual_drops = TestModel._values_from_config(config, 'SegMe>Common>DropPath', 'rate')
        self.assertListEqual(expected_drops, actual_drops)

    def test_residual_gamma(self):
        config = HardSwinTiny(
            embed_dim=64, stage_depths=(2, 2, 6, 4), pretrain_window=12, weights=None, include_top=False,
            input_shape=(None, None, 3)).get_config()

        expected_gammas = [
            ('stage_0_attn_0_swin_norm', 0.01), ('stage_0_attn_0_mlp_norm', 0.01),
            ('stage_0_attn_1_swin_norm', 0.009231538461538461), ('stage_0_attn_1_mlp_norm', 0.009231538461538461),

            ('stage_1_attn_0_swin_norm', 0.008463076923076924), ('stage_1_attn_0_mlp_norm', 0.008463076923076924),
            ('stage_1_attn_1_swin_norm', 0.007694615384615385), ('stage_1_attn_1_mlp_norm', 0.007694615384615385),

            ('stage_2_attn_0_swin_norm', 0.006926153846153846), ('stage_2_attn_0_mlp_norm', 0.006926153846153846),
            ('stage_2_attn_1_swin_norm', 0.006157692307692308), ('stage_2_attn_1_mlp_norm', 0.006157692307692308),
            ('stage_2_attn_2_swin_norm', 0.005389230769230769), ('stage_2_attn_2_mlp_norm', 0.005389230769230769),
            ('stage_2_attn_3_swin_norm', 0.00462076923076923), ('stage_2_attn_3_mlp_norm', 0.00462076923076923),
            ('stage_2_attn_4_swin_norm', 0.003852307692307692), ('stage_2_attn_4_mlp_norm', 0.003852307692307692),
            ('stage_2_attn_5_swin_norm', 0.003083846153846154), ('stage_2_attn_5_mlp_norm', 0.003083846153846154),

            ('stage_3_attn_0_swin_norm', 0.002315384615384615), ('stage_3_attn_0_mlp_norm', 0.002315384615384615),
            ('stage_3_attn_1_swin_norm', 0.001546923076923076), ('stage_3_attn_1_mlp_norm', 0.001546923076923076),
            ('stage_3_attn_2_swin_norm', 0.0007784615384615386), ('stage_3_attn_2_mlp_norm', 0.0007784615384615386),
            ('stage_3_attn_3_swin_norm', 1e-05), ('stage_3_attn_3_mlp_norm', 1e-05)]

        actual_gammas = TestModel._values_from_config(
            config, 'SegMe>Policy>Normalization>LayerNorm', 'gamma_initializer')
        actual_gammas = [(ag[0], ag[1]['config']['value']) for ag in actual_gammas if 'Constant' == ag[1]['class_name']]
        self.assertListEqual(expected_gammas, actual_gammas)

    def test_attention_shift(self):
        config = HardSwinTiny(
            embed_dim=64, stage_depths=(2, 2, 6, 4), pretrain_window=12, pretrain_size=384, weights=None,
            include_top=False, input_shape=(None, None, 3)).get_config()

        expected_shifts = [
            ('stage_0_attn_0_swin_attn', 0), ('stage_0_attn_1_swin_attn', 1),

            ('stage_1_attn_0_swin_attn', 0), ('stage_1_attn_1_swin_attn', 1),

            ('stage_2_attn_0_swin_attn', 0), ('stage_2_attn_1_swin_attn', 1),
            ('stage_2_attn_2_swin_attn', 0), ('stage_2_attn_3_swin_attn', 1),
            ('stage_2_attn_4_swin_attn', 0), ('stage_2_attn_5_swin_attn', 1),

            ('stage_3_attn_0_swin_attn', 0), ('stage_3_attn_1_swin_attn', 1),
            ('stage_3_attn_2_swin_attn', 0), ('stage_3_attn_3_swin_attn', 1)]

        actual_shifts = TestModel._values_from_config(config, 'SegMe>Common>SwinAttention', 'shift_mode')
        self.assertListEqual(expected_shifts, actual_shifts)

    def test_attention_window(self):
        config = HardSwinTiny(
            embed_dim=64, stage_depths=(2, 2, 6, 4), pretrain_window=16, pretrain_size=256, weights=None,
            include_top=False, input_shape=(None, None, 3)).get_config()

        expected_shifts = [
            ('stage_0_attn_0_swin_attn', 16), ('stage_0_attn_1_swin_attn', 16),

            ('stage_1_attn_0_swin_attn', 16), ('stage_1_attn_1_swin_attn', 16),

            ('stage_2_attn_0_swin_attn', 16), ('stage_2_attn_1_swin_attn', 16),
            ('stage_2_attn_2_swin_attn', 16), ('stage_2_attn_3_swin_attn', 16),
            ('stage_2_attn_4_swin_attn', 16), ('stage_2_attn_5_swin_attn', 16),

            ('stage_3_attn_0_swin_attn', 8), ('stage_3_attn_1_swin_attn', 8),
            ('stage_3_attn_2_swin_attn', 8), ('stage_3_attn_3_swin_attn', 8)]

        actual_shifts = TestModel._values_from_config(
            config, 'SegMe>Common>SwinAttention', 'current_window')
        self.assertListEqual(expected_shifts, actual_shifts)

    @parameterized.parameters((False,), (True,))
    def test_train(self, use_fp16):
        if use_fp16:
            mixed_precision.set_global_policy('mixed_float16')

        model = HardSwinTiny(embed_dim=64, stage_depths=(4, 4, 4, 4), pretrain_window=12, include_top=True,
                             weights=None)
        model.compile(optimizer='rmsprop', loss='mse', run_eagerly=test_utils.should_run_eagerly())

        images = np.random.random((10, 384, 384, 3)).astype('float32')
        labels = (np.random.random((10, 1)) + 0.5).astype('int64')
        model.fit(images, labels, epochs=1, batch_size=2)

        # test config
        model.get_config()

        # check whether the model variables are present in the trackable list of objects
        checkpointed_objects = object_identity.ObjectIdentitySet(checkpoint.list_objects(model))
        for v in model.variables:
            self.assertIn(v, checkpointed_objects)

    def test_finite(self):
        model = HardSwinTiny(
            embed_dim=64, stage_depths=(4, 4, 4, 4), pretrain_window=12, weights=None, include_top=False,
            input_shape=(None, None, 3))
        outputs = model(np.random.uniform(0., 255., [2, 384, 384, 3]).astype('float32'))
        outputs = self.evaluate(outputs)
        self.assertTrue(np.isfinite(outputs).all())

    def test_var_shape(self):
        model = HardSwinTiny(
            embed_dim=64, stage_depths=(4, 4, 4, 4), pretrain_window=12, weights=None, include_top=False,
            input_shape=(None, None, 3))
        run_eagerly = test_utils.should_run_eagerly()
        model.compile(optimizer='rmsprop', loss='mse', run_eagerly=run_eagerly, jit_compile=not run_eagerly)

        images = np.random.random((10, 512, 384, 3)).astype('float32')
        labels = (np.random.random((10, 16, 12, 512)) + 0.5).astype('int64')
        model.fit(images, labels, epochs=1, batch_size=2)

        # test config
        model.get_config()

        # check whether the model variables are present in the trackable list of objects
        checkpointed_objects = object_identity.ObjectIdentitySet(checkpoint.list_objects(model))
        for v in model.variables:
            self.assertIn(v, checkpointed_objects)

    def test_tiny(self):
        with cnapol.policy_scope('conv-ln1em5-gelu'):
            layer_multi_io_test(
                Backbone,
                kwargs={'scales': None, 'policy': 'hardswin_tiny-none'},
                input_shapes=[(2, 256, 256, 3)],
                input_dtypes=['uint8'],
                expected_output_shapes=[
                    (None, 64, 64, 96),
                    (None, 32, 32, 192),
                    (None, 16, 16, 384),
                    (None, 8, 8, 768)
                ],
                expected_output_dtypes=['float32'] * 4
            )

    def test_small_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        with cnapol.policy_scope('conv-ln1em5-gelu'):
            layer_multi_io_test(
                Backbone,
                kwargs={'scales': None, 'policy': 'hardswin_small-none'},
                input_shapes=[(2, 256, 256, 3)],
                input_dtypes=['uint8'],
                expected_output_shapes=[
                    (None, 64, 64, 96),
                    (None, 32, 32, 192),
                    (None, 16, 16, 384),
                    (None, 8, 8, 768)
                ],
                expected_output_dtypes=['float16'] * 4
            )

    def test_base(self):
        with cnapol.policy_scope('conv-ln1em5-gelu'):
            layer_multi_io_test(
                Backbone,
                kwargs={'scales': None, 'policy': 'hardswin_base-none'},
                input_shapes=[(2, 384, 384, 3)],
                input_dtypes=['uint8'],
                expected_output_shapes=[
                    (None, 96, 96, 128),
                    (None, 48, 48, 256),
                    (None, 24, 24, 512),
                    (None, 12, 12, 1024)
                ],
                expected_output_dtypes=['float32'] * 4
            )

    def test_large(self):
        with cnapol.policy_scope('conv-ln1em5-gelu'):
            layer_multi_io_test(
                Backbone,
                kwargs={'scales': None, 'policy': 'hardswin_large-none'},
                input_shapes=[(2, 384, 384, 3)],
                input_dtypes=['uint8'],
                expected_output_shapes=[
                    (None, 96, 96, 192),
                    (None, 48, 48, 384),
                    (None, 24, 24, 768),
                    (None, 12, 12, 1536)
                ],
                expected_output_dtypes=['float32'] * 4
            )


if __name__ == '__main__':
    tf.test.main()
