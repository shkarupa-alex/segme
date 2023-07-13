import numpy as np
import tensorflow as tf
from absl.testing import parameterized
from keras import mixed_precision
from keras.src.testing_infra import test_combinations, test_utils
from tensorflow.python.util import object_identity
from tensorflow.python.training.tracking import util as trackable_util
from segme.policy.backbone.diy.softswin.model import SoftSwin


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
        config = SoftSwin(
            embed_dim=64, stem_depth=2, stage_depths=(2, 2, 6, 4), pretrain_window=12, weights=None, include_top=False,
            input_shape=(None, None, 3)).get_config()

        expected_drops = [
            ('stem_1_drop', 0.0), ('stem_2_drop', 0.013333333333333334),

            ('stage_0_attn_0_swin_drop', 0.02666666666666667), ('stage_0_attn_0_mlp_drop', 0.02666666666666667),
            ('stage_0_attn_1_swin_drop', 0.04), ('stage_0_attn_1_mlp_drop', 0.04),

            ('stage_1_attn_0_swin_drop', 0.05333333333333334), ('stage_1_attn_0_mlp_drop', 0.05333333333333334),
            ('stage_1_attn_1_swin_drop', 0.06666666666666667), ('stage_1_attn_1_mlp_drop', 0.06666666666666667),

            ('stage_2_attn_0_swin_drop', 0.08), ('stage_2_attn_0_mlp_drop', 0.08),
            ('stage_2_attn_1_swin_drop', 0.09333333333333334), ('stage_2_attn_1_mlp_drop', 0.09333333333333334),
            ('stage_2_attn_2_swin_drop', 0.10666666666666667), ('stage_2_attn_2_mlp_drop', 0.10666666666666667),
            ('stage_2_attn_3_swin_drop', 0.12000000000000001), ('stage_2_attn_3_mlp_drop', 0.12000000000000001),
            ('stage_2_attn_4_swin_drop', 0.13333333333333333), ('stage_2_attn_4_mlp_drop', 0.13333333333333333),
            ('stage_2_attn_5_swin_drop', 0.14666666666666667), ('stage_2_attn_5_mlp_drop', 0.14666666666666667),

            ('stage_3_attn_0_swin_drop', 0.16), ('stage_3_attn_0_mlp_drop', 0.16),
            ('stage_3_attn_1_swin_drop', 0.17333333333333334), ('stage_3_attn_1_mlp_drop', 0.17333333333333334),
            ('stage_3_attn_2_swin_drop', 0.18666666666666668), ('stage_3_attn_2_mlp_drop', 0.18666666666666668),
            ('stage_3_attn_3_swin_drop', 0.2), ('stage_3_attn_3_mlp_drop', 0.2)]

        actual_drops = TestModel._values_from_config(config, 'SegMe>Common>DropPath', 'rate')
        self.assertListEqual(expected_drops, actual_drops)

    def test_residual_gamma(self):
        config = SoftSwin(
            embed_dim=64, stem_depth=2, stage_depths=(2, 2, 6, 4), pretrain_window=12, weights=None, include_top=False,
            input_shape=(None, None, 3)).get_config()

        expected_gammas = [
            ('stem_1_norm', 0.01), ('stem_2_norm', 0.009334), ('stage_0_attn_0_swin_norm', 0.008668),

            ('stage_0_attn_0_mlp_norm', 0.008668), ('stage_0_attn_1_swin_norm', 0.008002),
            ('stage_0_attn_1_mlp_norm', 0.008002), ('stage_1_attn_0_swin_norm', 0.0073360000000000005),

            ('stage_1_attn_0_mlp_norm', 0.0073360000000000005), ('stage_1_attn_1_swin_norm', 0.006670000000000001),
            ('stage_1_attn_1_mlp_norm', 0.006670000000000001), ('stage_2_attn_0_swin_norm', 0.006004),
            ('stage_2_attn_0_mlp_norm', 0.006004), ('stage_2_attn_1_swin_norm', 0.005338),

            ('stage_2_attn_1_mlp_norm', 0.005338), ('stage_2_attn_2_swin_norm', 0.004672),
            ('stage_2_attn_2_mlp_norm', 0.004672), ('stage_2_attn_3_swin_norm', 0.004006),
            ('stage_2_attn_3_mlp_norm', 0.004006), ('stage_2_attn_4_swin_norm', 0.00334),
            ('stage_2_attn_4_mlp_norm', 0.00334), ('stage_2_attn_5_swin_norm', 0.002674),
            ('stage_2_attn_5_mlp_norm', 0.002674), ('stage_3_attn_0_swin_norm', 0.0020079999999999994),

            ('stage_3_attn_0_mlp_norm', 0.0020079999999999994),
            ('stage_3_attn_1_swin_norm', 0.0013419999999999994),
            ('stage_3_attn_1_mlp_norm', 0.0013419999999999994),
            ('stage_3_attn_2_swin_norm', 0.0006759999999999995),
            ('stage_3_attn_2_mlp_norm', 0.0006759999999999995), ('stage_3_attn_3_swin_norm', 1e-05),
            ('stage_3_attn_3_mlp_norm', 1e-05)]

        actual_gammas = TestModel._values_from_config(
            config, ['SegMe>Policy>Normalization>GroupNorm', 'SegMe>Policy>Normalization>LayerNorm'],
            'gamma_initializer')
        actual_gammas = [(ag[0], ag[1]['config']['value']) for ag in actual_gammas if 'Constant' == ag[1]['class_name']]
        self.assertListEqual(expected_gammas, actual_gammas)

    def test_attention_shift(self):
        config = SoftSwin(
            embed_dim=64, stem_depth=2, stage_depths=(2, 2, 6, 4), pretrain_window=12, pretrain_size=384, weights=None,
            include_top=False, input_shape=(None, None, 3)).get_config()

        expected_shifts = [
            ('stage_0_attn_0_swin_attn', 0), ('stage_0_attn_1_swin_attn', 1),

            ('stage_1_attn_0_swin_attn', 0), ('stage_1_attn_1_swin_attn', 2),

            ('stage_2_attn_0_swin_attn', 0), ('stage_2_attn_1_swin_attn', 3),
            ('stage_2_attn_2_swin_attn', 0), ('stage_2_attn_3_swin_attn', 4),
            ('stage_2_attn_4_swin_attn', 0), ('stage_2_attn_5_swin_attn', 1),

            ('stage_3_attn_0_swin_attn', 0), ('stage_3_attn_1_swin_attn', 2),
            ('stage_3_attn_2_swin_attn', 0), ('stage_3_attn_3_swin_attn', 3)]

        actual_shifts = TestModel._values_from_config(
            config, 'SegMe>Common>SwinAttention', 'shift_mode')
        self.assertListEqual(expected_shifts, actual_shifts)

    def test_attention_window(self):
        config = SoftSwin(
            embed_dim=64, stem_depth=2, stage_depths=(2, 2, 6, 4), pretrain_window=16, pretrain_size=256, weights=None,
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

        model = SoftSwin(embed_dim=64, stem_depth=2, stage_depths=(4, 4, 4, 4), pretrain_window=12, weights=None)
        model.compile(optimizer='rmsprop', loss='mse', run_eagerly=test_utils.should_run_eagerly())

        images = np.random.random((10, 384, 384, 3)).astype('float32')
        labels = (np.random.random((10, 1)) + 0.5).astype('int64')
        model.fit(images, labels, epochs=1, batch_size=2)

        # test config
        model.get_config()

        # check whether the model variables are present in the trackable list of objects
        checkpointed_objects = object_identity.ObjectIdentitySet(trackable_util.list_objects(model))
        for v in model.variables:
            self.assertIn(v, checkpointed_objects)

    def test_finite(self):
        model = SoftSwin(
            embed_dim=64, stem_depth=2, stage_depths=(4, 4, 4, 4), pretrain_window=12, weights=None, include_top=False,
            input_shape=(None, None, 3))
        outputs = model(np.random.uniform(0., 255., [2, 384, 384, 3]).astype('float32'))
        outputs = self.evaluate(outputs)
        self.assertTrue(np.isfinite(outputs).all())

    def test_var_shape(self):
        model = SoftSwin(
            embed_dim=64, stem_depth=2, stage_depths=(4, 4, 4, 4), pretrain_window=12, weights=None, include_top=False,
            input_shape=(None, None, 3))
        run_eagerly = test_utils.should_run_eagerly()
        model.compile(optimizer='rmsprop', loss='mse', run_eagerly=run_eagerly, jit_compile=not run_eagerly)

        images = np.random.random((10, 512, 384, 3)).astype('float32')
        labels = (np.random.random((10, 16, 12, 512)) + 0.5).astype('int64')
        model.fit(images, labels, epochs=1, batch_size=2)

        # test config
        model.get_config()

        # check whether the model variables are present in the trackable list of objects
        checkpointed_objects = object_identity.ObjectIdentitySet(trackable_util.list_objects(model))
        for v in model.variables:
            self.assertIn(v, checkpointed_objects)


if __name__ == '__main__':
    tf.test.main()
