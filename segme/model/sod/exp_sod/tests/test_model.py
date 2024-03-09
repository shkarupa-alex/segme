import numpy as np
import tensorflow as tf
from keras import mixed_precision
from keras.src.testing_infra import test_combinations, test_utils
from tensorflow.python.checkpoint import checkpoint
from tensorflow.python.util import object_identity
from segme.model.sod.exp_sod.model import ExpSOD
from segme.model.sod.exp_sod.loss import exp_sod_losses
from segme.testing_utils import layer_multi_io_test


# @test_combinations.run_all_keras_modes
class TestExpSOD(test_combinations.TestCase):
    def setUp(self):
        super(TestExpSOD, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestExpSOD, self).tearDown()
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
            values = sorted(values, key=lambda x: x[0])

        return values

    def test_drop_path(self):
        config = ExpSOD().get_config()

        expected_drops = [
            ('backstage_1_lateral_transform_0_mlp_drop', 0.2), ('backstage_1_lateral_transform_0_swin_drop', 0.2),
            ('backstage_1_lateral_transform_1_mlp_drop', 0.18666666666666668),
            ('backstage_1_lateral_transform_1_swin_drop', 0.18666666666666668),
            ('backstage_1_merge_transform_0_mlp_drop', 0.17333333333333334),
            ('backstage_1_merge_transform_0_swin_drop', 0.17333333333333334),
            ('backstage_1_merge_transform_1_mlp_drop', 0.16), ('backstage_1_merge_transform_1_swin_drop', 0.16),

            ('backstage_2_lateral_transform_0_mlp_drop', 0.14666666666666667),
            ('backstage_2_lateral_transform_0_swin_drop', 0.14666666666666667),
            ('backstage_2_lateral_transform_1_mlp_drop', 0.13333333333333336),
            ('backstage_2_lateral_transform_1_swin_drop', 0.13333333333333336),
            ('backstage_2_merge_transform_0_mlp_drop', 0.12000000000000001),
            ('backstage_2_merge_transform_0_swin_drop', 0.12000000000000001),
            ('backstage_2_merge_transform_1_mlp_drop', 0.10666666666666667),
            ('backstage_2_merge_transform_1_swin_drop', 0.10666666666666667),

            ('backstage_3_lateral_transform_0_mlp_drop', 0.09333333333333334),
            ('backstage_3_lateral_transform_0_swin_drop', 0.09333333333333334),
            ('backstage_3_lateral_transform_1_mlp_drop', 0.08), ('backstage_3_lateral_transform_1_swin_drop', 0.08),
            ('backstage_3_merge_transform_0_mlp_drop', 0.06666666666666668),
            ('backstage_3_merge_transform_0_swin_drop', 0.06666666666666668),
            ('backstage_3_merge_transform_1_mlp_drop', 0.053333333333333344),
            ('backstage_3_merge_transform_1_swin_drop', 0.053333333333333344),

            ('backstage_4_lateral_transform_0_mlp_drop', 0.04000000000000001),
            ('backstage_4_lateral_transform_0_swin_drop', 0.04000000000000001),
            ('backstage_4_lateral_transform_1_mlp_drop', 0.026666666666666672),
            ('backstage_4_lateral_transform_1_swin_drop', 0.026666666666666672),
            ('backstage_4_merge_transform_0_mlp_drop', 0.013333333333333336),
            ('backstage_4_merge_transform_0_swin_drop', 0.013333333333333336),
            ('backstage_4_merge_transform_1_mlp_drop', 0.0), ('backstage_4_merge_transform_1_swin_drop', 0.0)
        ]

        actual_drops = TestExpSOD._values_from_config(config, 'SegMe>Common>DropPath', 'rate')
        self.assertListEqual(expected_drops, actual_drops)

    def test_residual_gamma(self):
        config = ExpSOD().get_config()

        expected_gammas = [
            ('backstage_1_lateral_transform_0_mlp_norm', 1e-05), ('backstage_1_lateral_transform_0_swin_norm', 1e-05),
            ('backstage_1_lateral_transform_1_mlp_norm', 0.0066760000000000005),
            ('backstage_1_lateral_transform_1_swin_norm', 0.0066760000000000005),
            ('backstage_1_merge_transform_0_mlp_norm', 0.013342000000000001),
            ('backstage_1_merge_transform_0_swin_norm', 0.013342000000000001),
            ('backstage_1_merge_transform_1_mlp_norm', 0.020008), ('backstage_1_merge_transform_1_swin_norm', 0.020008),

            ('backstage_2_lateral_transform_0_mlp_norm', 0.026674000000000003),
            ('backstage_2_lateral_transform_0_swin_norm', 0.026674000000000003),
            ('backstage_2_lateral_transform_1_mlp_norm', 0.03334000000000001),
            ('backstage_2_lateral_transform_1_swin_norm', 0.03334000000000001),
            ('backstage_2_merge_transform_0_mlp_norm', 0.04000600000000001),
            ('backstage_2_merge_transform_0_swin_norm', 0.04000600000000001),
            ('backstage_2_merge_transform_1_mlp_norm', 0.04667200000000001),
            ('backstage_2_merge_transform_1_swin_norm', 0.04667200000000001),

            ('backstage_3_lateral_transform_0_mlp_norm', 0.05333800000000001),
            ('backstage_3_lateral_transform_0_swin_norm', 0.05333800000000001),
            ('backstage_3_lateral_transform_1_mlp_norm', 0.06000400000000001),
            ('backstage_3_lateral_transform_1_swin_norm', 0.06000400000000001),
            ('backstage_3_merge_transform_0_mlp_norm', 0.06667000000000001),
            ('backstage_3_merge_transform_0_swin_norm', 0.06667000000000001),
            ('backstage_3_merge_transform_1_mlp_norm', 0.07333600000000001),
            ('backstage_3_merge_transform_1_swin_norm', 0.07333600000000001),

            ('backstage_4_lateral_transform_0_mlp_norm', 0.080002),
            ('backstage_4_lateral_transform_0_swin_norm', 0.080002),
            ('backstage_4_lateral_transform_1_mlp_norm', 0.08666800000000001),
            ('backstage_4_lateral_transform_1_swin_norm', 0.08666800000000001),
            ('backstage_4_merge_transform_0_mlp_norm', 0.09333400000000001),
            ('backstage_4_merge_transform_0_swin_norm', 0.09333400000000001),
            ('backstage_4_merge_transform_1_mlp_norm', 0.1), ('backstage_4_merge_transform_1_swin_norm', 0.1)
        ]

        actual_gammas = TestExpSOD._values_from_config(
            config, 'SegMe>Policy>Normalization>LayerNorm', 'gamma_initializer')
        actual_gammas = [(ag[0], ag[1]['config']['value']) for ag in actual_gammas if 'Constant' == ag[1]['class_name']]
        self.assertListEqual(expected_gammas, actual_gammas)

    def test_attention_shift(self):
        config = ExpSOD().get_config()

        expected_shifts = [
            ('backstage_1_lateral_transform_0_swin_attn', 0), ('backstage_1_lateral_transform_1_swin_attn', 1),
            ('backstage_1_merge_transform_0_swin_attn', 0), ('backstage_1_merge_transform_1_swin_attn', 2),

            ('backstage_2_lateral_transform_0_swin_attn', 0), ('backstage_2_lateral_transform_1_swin_attn', 3),
            ('backstage_2_merge_transform_0_swin_attn', 0), ('backstage_2_merge_transform_1_swin_attn', 4),

            ('backstage_3_lateral_transform_0_swin_attn', 0), ('backstage_3_lateral_transform_1_swin_attn', 1),
            ('backstage_3_merge_transform_0_swin_attn', 0), ('backstage_3_merge_transform_1_swin_attn', 2),

            ('backstage_4_lateral_transform_0_swin_attn', 0), ('backstage_4_lateral_transform_1_swin_attn', 3),
            ('backstage_4_merge_transform_0_swin_attn', 0), ('backstage_4_merge_transform_1_swin_attn', 4)
        ]

        actual_shifts = TestExpSOD._values_from_config(
            config, 'SegMe>Common>SwinAttention', 'shift_mode')
        self.assertListEqual(expected_shifts, actual_shifts)

    def test_attention_window(self):
        config = ExpSOD().get_config()

        expected_shifts = [
            ('backstage_1_lateral_transform_0_swin_attn', 24), ('backstage_1_lateral_transform_1_swin_attn', 24),
            ('backstage_1_merge_transform_0_swin_attn', 24), ('backstage_1_merge_transform_1_swin_attn', 24),

            ('backstage_2_lateral_transform_0_swin_attn', 24), ('backstage_2_lateral_transform_1_swin_attn', 24),
            ('backstage_2_merge_transform_0_swin_attn', 24), ('backstage_2_merge_transform_1_swin_attn', 24),

            ('backstage_3_lateral_transform_0_swin_attn', 24), ('backstage_3_lateral_transform_1_swin_attn', 24),
            ('backstage_3_merge_transform_0_swin_attn', 24), ('backstage_3_merge_transform_1_swin_attn', 24),

            ('backstage_4_lateral_transform_0_swin_attn', 24), ('backstage_4_lateral_transform_1_swin_attn', 24),
            ('backstage_4_merge_transform_0_swin_attn', 24), ('backstage_4_merge_transform_1_swin_attn', 24)
        ]

        actual_shifts = TestExpSOD._values_from_config(
            config, 'SegMe>Common>SwinAttention', 'current_window')
        self.assertListEqual(expected_shifts, actual_shifts)

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
