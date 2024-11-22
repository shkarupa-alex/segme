import numpy as np
from keras.src import testing

from segme.model.matting.exp_mat.loss import exp_mat_losses
from segme.model.matting.exp_mat.model import ExpMat


class TestExpMat(testing.TestCase):
    @staticmethod
    def _values_from_config(cfg, cls, prop):
        if isinstance(cls, str):
            cls = [cls]

        values = []
        for layer in cfg["layers"]:
            if layer["registered_name"] not in cls:
                continue

            if prop not in layer["config"]:
                continue

            values.append((layer["name"], layer["config"][prop]))
            values = sorted(values, key=lambda x: x[0])

        return values

    def test_drop_path(self):
        config = ExpMat().get_config()

        expected_drops = [
            ("backstage_1_lateral_transform_0_mlp_drop", 0.2),
            ("backstage_1_lateral_transform_0_swin_drop", 0.2),
            ("backstage_1_lateral_transform_1_mlp_drop", 0.18666666666666668),
            ("backstage_1_lateral_transform_1_swin_drop", 0.18666666666666668),
            ("backstage_1_merge_transform_0_mlp_drop", 0.17333333333333334),
            ("backstage_1_merge_transform_0_swin_drop", 0.17333333333333334),
            ("backstage_1_merge_transform_1_mlp_drop", 0.16),
            ("backstage_1_merge_transform_1_swin_drop", 0.16),
            ("backstage_2_lateral_transform_0_mlp_drop", 0.14666666666666667),
            ("backstage_2_lateral_transform_0_swin_drop", 0.14666666666666667),
            ("backstage_2_lateral_transform_1_mlp_drop", 0.13333333333333336),
            ("backstage_2_lateral_transform_1_swin_drop", 0.13333333333333336),
            ("backstage_2_merge_transform_0_mlp_drop", 0.12000000000000001),
            ("backstage_2_merge_transform_0_swin_drop", 0.12000000000000001),
            ("backstage_2_merge_transform_1_mlp_drop", 0.10666666666666667),
            ("backstage_2_merge_transform_1_swin_drop", 0.10666666666666667),
            ("backstage_3_lateral_transform_0_mlp_drop", 0.09333333333333334),
            ("backstage_3_lateral_transform_0_swin_drop", 0.09333333333333334),
            ("backstage_3_lateral_transform_1_mlp_drop", 0.08),
            ("backstage_3_lateral_transform_1_swin_drop", 0.08),
            ("backstage_3_merge_transform_0_mlp_drop", 0.06666666666666668),
            ("backstage_3_merge_transform_0_swin_drop", 0.06666666666666668),
            ("backstage_3_merge_transform_1_mlp_drop", 0.053333333333333344),
            ("backstage_3_merge_transform_1_swin_drop", 0.053333333333333344),
            (
                "backstage_4_lateral_transform_0_fmbconv_drop",
                0.04000000000000001,
            ),
            (
                "backstage_4_lateral_transform_1_fmbconv_drop",
                0.04000000000000001,
            ),
            (
                "backstage_4_lateral_transform_2_fmbconv_drop",
                0.026666666666666672,
            ),
            (
                "backstage_4_lateral_transform_3_fmbconv_drop",
                0.026666666666666672,
            ),
            (
                "backstage_4_merge_transform_0_fmbconv_drop",
                0.013333333333333336,
            ),
            (
                "backstage_4_merge_transform_1_fmbconv_drop",
                0.013333333333333336,
            ),
            ("backstage_4_merge_transform_2_fmbconv_drop", 0.0),
            ("backstage_4_merge_transform_3_fmbconv_drop", 0.0),
        ]

        actual_drops = TestExpMat._values_from_config(
            config, "SegMe>Common>DropPath", "rate"
        )
        self.assertListEqual(expected_drops, actual_drops)

    def test_residual_gamma(self):
        config = ExpMat().get_config()

        expected_gammas = [
            ("backstage_1_lateral_transform_0_mlp_norm", 1e-05),
            ("backstage_1_lateral_transform_0_swin_norm", 1e-05),
            ("backstage_1_lateral_transform_1_mlp_norm", 0.0066760000000000005),
            (
                "backstage_1_lateral_transform_1_swin_norm",
                0.0066760000000000005,
            ),
            ("backstage_1_merge_transform_0_mlp_norm", 0.013342000000000001),
            ("backstage_1_merge_transform_0_swin_norm", 0.013342000000000001),
            ("backstage_1_merge_transform_1_mlp_norm", 0.020008),
            ("backstage_1_merge_transform_1_swin_norm", 0.020008),
            ("backstage_2_lateral_transform_0_mlp_norm", 0.026674000000000003),
            ("backstage_2_lateral_transform_0_swin_norm", 0.026674000000000003),
            ("backstage_2_lateral_transform_1_mlp_norm", 0.03334000000000001),
            ("backstage_2_lateral_transform_1_swin_norm", 0.03334000000000001),
            ("backstage_2_merge_transform_0_mlp_norm", 0.04000600000000001),
            ("backstage_2_merge_transform_0_swin_norm", 0.04000600000000001),
            ("backstage_2_merge_transform_1_mlp_norm", 0.04667200000000001),
            ("backstage_2_merge_transform_1_swin_norm", 0.04667200000000001),
            ("backstage_3_lateral_transform_0_mlp_norm", 0.05333800000000001),
            ("backstage_3_lateral_transform_0_swin_norm", 0.05333800000000001),
            ("backstage_3_lateral_transform_1_mlp_norm", 0.06000400000000001),
            ("backstage_3_lateral_transform_1_swin_norm", 0.06000400000000001),
            ("backstage_3_merge_transform_0_mlp_norm", 0.06667000000000001),
            ("backstage_3_merge_transform_0_swin_norm", 0.06667000000000001),
            ("backstage_3_merge_transform_1_mlp_norm", 0.07333600000000001),
            ("backstage_3_merge_transform_1_swin_norm", 0.07333600000000001),
            ("backstage_4_lateral_transform_0_fmbconv_norm", 0.080002),
            ("backstage_4_lateral_transform_1_fmbconv_norm", 0.080002),
            (
                "backstage_4_lateral_transform_2_fmbconv_norm",
                0.08666800000000001,
            ),
            (
                "backstage_4_lateral_transform_3_fmbconv_norm",
                0.08666800000000001,
            ),
            ("backstage_4_merge_transform_0_fmbconv_norm", 0.09333400000000001),
            ("backstage_4_merge_transform_1_fmbconv_norm", 0.09333400000000001),
            ("backstage_4_merge_transform_2_fmbconv_norm", 0.1),
            ("backstage_4_merge_transform_3_fmbconv_norm", 0.1),
        ]

        actual_gammas = TestExpMat._values_from_config(
            config, "SegMe>Policy>Normalization>LayerNorm", "gamma_initializer"
        )
        actual_gammas = [
            (ag[0], ag[1]["config"]["value"])
            for ag in actual_gammas
            if "Constant" == ag[1]["class_name"]
        ]
        self.assertListEqual(expected_gammas, actual_gammas)

    def test_attention_shift(self):
        config = ExpMat().get_config()

        expected_shifts = [
            ("backstage_1_lateral_transform_0_swin_attn", 0),
            ("backstage_1_lateral_transform_1_swin_attn", 1),
            ("backstage_1_merge_transform_0_swin_attn", 0),
            ("backstage_1_merge_transform_1_swin_attn", 2),
            ("backstage_2_lateral_transform_0_swin_attn", 0),
            ("backstage_2_lateral_transform_1_swin_attn", 3),
            ("backstage_2_merge_transform_0_swin_attn", 0),
            ("backstage_2_merge_transform_1_swin_attn", 4),
            ("backstage_3_lateral_transform_0_swin_attn", 0),
            ("backstage_3_lateral_transform_1_swin_attn", 1),
            ("backstage_3_merge_transform_0_swin_attn", 0),
            ("backstage_3_merge_transform_1_swin_attn", 2),
        ]

        actual_shifts = TestExpMat._values_from_config(
            config, "SegMe>Common>SwinAttention", "shift_mode"
        )
        self.assertListEqual(expected_shifts, actual_shifts)

    def test_attention_window(self):
        config = ExpMat().get_config()

        expected_shifts = [
            ("backstage_1_lateral_transform_0_swin_attn", 24),
            ("backstage_1_lateral_transform_1_swin_attn", 24),
            ("backstage_1_merge_transform_0_swin_attn", 24),
            ("backstage_1_merge_transform_1_swin_attn", 24),
            ("backstage_2_lateral_transform_0_swin_attn", 24),
            ("backstage_2_lateral_transform_1_swin_attn", 24),
            ("backstage_2_merge_transform_0_swin_attn", 24),
            ("backstage_2_merge_transform_1_swin_attn", 24),
            ("backstage_3_lateral_transform_0_swin_attn", 24),
            ("backstage_3_lateral_transform_1_swin_attn", 24),
            ("backstage_3_merge_transform_0_swin_attn", 24),
            ("backstage_3_merge_transform_1_swin_attn", 24),
        ]

        actual_shifts = TestExpMat._values_from_config(
            config, "SegMe>Common>SwinAttention", "current_window"
        )
        self.assertListEqual(expected_shifts, actual_shifts)

    def test_layer(self):
        self.run_layer_test(
            ExpMat,
            init_kwargs={
                "transform_depth": 2,
                "window_size": 24,
                "path_gamma": 0.01,
                "path_drop": 0.2,
            },
            input_shape=((2, 384, 384, 3), (2, 384, 384, 1)),
            input_dtype=("uint8",) * 2,
            expected_output_shape=((2, 384, 384, 7),) * 5 + ((2, 384, 384, 1),),
            expected_output_dtype=("float32",) * 6,
        )

    def test_model(self):
        model = ExpMat()
        model.compile(
            optimizer="sgd",
            loss=exp_mat_losses(5),
        )
        model.fit(
            [
                np.random.random((2, 384, 384, 3)).astype("uint8"),
                np.random.random((2, 384, 384, 1)).astype("uint8"),
            ],
            [
                np.random.random((2, 384, 384, 7)).astype("float32"),
            ]
            * 5
            + [
                np.random.random((2, 384, 384, 1)).astype("float32"),
            ],
            epochs=1,
            batch_size=10,
        )

        # test config
        model.get_config()
