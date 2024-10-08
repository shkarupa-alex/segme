import numpy as np
from keras.src import testing

from segme.common.backbone import Backbone
from segme.policy.backbone.diy.softswin import SoftSwinTiny


class TestModel(testing.TestCase):
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

        return values

    def test_drop_path(self):
        config = SoftSwinTiny(
            embed_dim=64,
            stage_depths=(2, 2, 6, 4),
            pretrain_window=12,
            weights=None,
            include_top=False,
            input_shape=(None, None, 3),
        ).get_config()

        expected_drops = [
            ("stage_0_attn_0_swin_drop", 0.0),
            ("stage_0_attn_0_mlp_drop", 0.0),
            ("stage_0_attn_1_swin_drop", 0.015384615384615385),
            ("stage_0_attn_1_mlp_drop", 0.015384615384615385),
            ("stage_1_attn_0_swin_drop", 0.03076923076923077),
            ("stage_1_attn_0_mlp_drop", 0.03076923076923077),
            ("stage_1_attn_1_swin_drop", 0.046153846153846156),
            ("stage_1_attn_1_mlp_drop", 0.046153846153846156),
            ("stage_2_attn_0_swin_drop", 0.06153846153846154),
            ("stage_2_attn_0_mlp_drop", 0.06153846153846154),
            ("stage_2_attn_1_swin_drop", 0.07692307692307693),
            ("stage_2_attn_1_mlp_drop", 0.07692307692307693),
            ("stage_2_attn_2_swin_drop", 0.09230769230769231),
            ("stage_2_attn_2_mlp_drop", 0.09230769230769231),
            ("stage_2_attn_3_swin_drop", 0.1076923076923077),
            ("stage_2_attn_3_mlp_drop", 0.1076923076923077),
            ("stage_2_attn_4_swin_drop", 0.12307692307692308),
            ("stage_2_attn_4_mlp_drop", 0.12307692307692308),
            ("stage_2_attn_5_swin_drop", 0.13846153846153847),
            ("stage_2_attn_5_mlp_drop", 0.13846153846153847),
            ("stage_3_attn_0_swin_drop", 0.15384615384615385),
            ("stage_3_attn_0_mlp_drop", 0.15384615384615385),
            ("stage_3_attn_1_swin_drop", 0.16923076923076924),
            ("stage_3_attn_1_mlp_drop", 0.16923076923076924),
            ("stage_3_attn_2_swin_drop", 0.18461538461538463),
            ("stage_3_attn_2_mlp_drop", 0.18461538461538463),
            ("stage_3_attn_3_swin_drop", 0.2),
            ("stage_3_attn_3_mlp_drop", 0.2),
        ]

        actual_drops = TestModel._values_from_config(
            config, "SegMe>Common>DropPath", "rate"
        )
        self.assertListEqual(expected_drops, actual_drops)

    def test_residual_gamma(self):
        config = SoftSwinTiny(
            embed_dim=64,
            stage_depths=(2, 2, 6, 4),
            pretrain_window=12,
            weights=None,
            include_top=False,
            input_shape=(None, None, 3),
        ).get_config()

        expected_gammas = [
            ("stage_0_attn_0_swin_norm", 0.01),
            ("stage_0_attn_0_mlp_norm", 0.01),
            ("stage_0_attn_1_swin_norm", 0.009231538461538461),
            ("stage_0_attn_1_mlp_norm", 0.009231538461538461),
            ("stage_1_attn_0_swin_norm", 0.008463076923076924),
            ("stage_1_attn_0_mlp_norm", 0.008463076923076924),
            ("stage_1_attn_1_swin_norm", 0.007694615384615385),
            ("stage_1_attn_1_mlp_norm", 0.007694615384615385),
            ("stage_2_attn_0_swin_norm", 0.006926153846153846),
            ("stage_2_attn_0_mlp_norm", 0.006926153846153846),
            ("stage_2_attn_1_swin_norm", 0.006157692307692308),
            ("stage_2_attn_1_mlp_norm", 0.006157692307692308),
            ("stage_2_attn_2_swin_norm", 0.005389230769230769),
            ("stage_2_attn_2_mlp_norm", 0.005389230769230769),
            ("stage_2_attn_3_swin_norm", 0.00462076923076923),
            ("stage_2_attn_3_mlp_norm", 0.00462076923076923),
            ("stage_2_attn_4_swin_norm", 0.003852307692307692),
            ("stage_2_attn_4_mlp_norm", 0.003852307692307692),
            ("stage_2_attn_5_swin_norm", 0.003083846153846154),
            ("stage_2_attn_5_mlp_norm", 0.003083846153846154),
            ("stage_3_attn_0_swin_norm", 0.002315384615384615),
            ("stage_3_attn_0_mlp_norm", 0.002315384615384615),
            ("stage_3_attn_1_swin_norm", 0.001546923076923076),
            ("stage_3_attn_1_mlp_norm", 0.001546923076923076),
            ("stage_3_attn_2_swin_norm", 0.0007784615384615386),
            ("stage_3_attn_2_mlp_norm", 0.0007784615384615386),
            ("stage_3_attn_3_swin_norm", 1e-05),
            ("stage_3_attn_3_mlp_norm", 1e-05),
        ]

        actual_gammas = TestModel._values_from_config(
            config, "SegMe>Policy>Normalization>LayerNorm", "gamma_initializer"
        )
        actual_gammas = [
            (ag[0], ag[1]["config"]["value"])
            for ag in actual_gammas
            if "Constant" == ag[1]["class_name"]
        ]
        self.assertListEqual(expected_gammas, actual_gammas)

    def test_attention_shift(self):
        config = SoftSwinTiny(
            embed_dim=64,
            stage_depths=(2, 2, 6, 4),
            pretrain_window=12,
            pretrain_size=384,
            weights=None,
            include_top=False,
            input_shape=(None, None, 3),
        ).get_config()

        expected_shifts = [
            ("stage_0_attn_0_swin_attn", 0),
            ("stage_0_attn_1_swin_attn", 1),
            ("stage_1_attn_0_swin_attn", 0),
            ("stage_1_attn_1_swin_attn", 2),
            ("stage_2_attn_0_swin_attn", 0),
            ("stage_2_attn_1_swin_attn", 3),
            ("stage_2_attn_2_swin_attn", 0),
            ("stage_2_attn_3_swin_attn", 4),
            ("stage_2_attn_4_swin_attn", 0),
            ("stage_2_attn_5_swin_attn", 1),
            ("stage_3_attn_0_swin_attn", 0),
            ("stage_3_attn_1_swin_attn", 2),
            ("stage_3_attn_2_swin_attn", 0),
            ("stage_3_attn_3_swin_attn", 3),
        ]

        actual_shifts = TestModel._values_from_config(
            config, "SegMe>Common>SwinAttention", "shift_mode"
        )
        self.assertListEqual(expected_shifts, actual_shifts)

    def test_attention_window(self):
        config = SoftSwinTiny(
            embed_dim=64,
            stage_depths=(2, 2, 6, 4),
            pretrain_window=16,
            pretrain_size=256,
            weights=None,
            include_top=False,
            input_shape=(None, None, 3),
        ).get_config()

        expected_shifts = [
            ("stage_0_attn_0_swin_attn", 16),
            ("stage_0_attn_1_swin_attn", 16),
            ("stage_1_attn_0_swin_attn", 16),
            ("stage_1_attn_1_swin_attn", 16),
            ("stage_2_attn_0_swin_attn", 16),
            ("stage_2_attn_1_swin_attn", 16),
            ("stage_2_attn_2_swin_attn", 16),
            ("stage_2_attn_3_swin_attn", 16),
            ("stage_2_attn_4_swin_attn", 16),
            ("stage_2_attn_5_swin_attn", 16),
            ("stage_3_attn_0_swin_attn", 8),
            ("stage_3_attn_1_swin_attn", 8),
            ("stage_3_attn_2_swin_attn", 8),
            ("stage_3_attn_3_swin_attn", 8),
        ]

        actual_shifts = TestModel._values_from_config(
            config, "SegMe>Common>SwinAttention", "current_window"
        )
        self.assertListEqual(expected_shifts, actual_shifts)

    def test_finite(self):
        model = SoftSwinTiny(
            embed_dim=64,
            stage_depths=(4, 4, 4, 4),
            pretrain_window=12,
            weights=None,
            include_top=False,
            input_shape=(None, None, 3),
        )
        outputs = model(
            np.random.uniform(0.0, 255.0, [2, 384, 384, 3]).astype("float32")
        )
        self.assertTrue(np.isfinite(outputs).all())

    def test_var_shape(self):
        model = SoftSwinTiny(
            embed_dim=64,
            stage_depths=(4, 4, 4, 4),
            pretrain_window=12,
            weights=None,
            include_top=False,
            input_shape=(None, None, 3),
        )
        model.compile(
            optimizer="rmsprop",
            loss="mse",
        )

        images = np.random.random((10, 512, 384, 3)).astype("float32")
        labels = (np.random.random((10, 16, 12, 512)) + 0.5).astype("int64")
        model.fit(images, labels, epochs=1, batch_size=2)

        # test config
        model.get_config()

    def test_tiny(self):
        self.run_layer_test(
            Backbone,
            init_kwargs={"scales": None, "policy": "softswin_tiny-none"},
            input_shape=(2, 256, 256, 3),
            input_dtype="uint8",
            expected_output_shape=(
                (2, 128, 128, 48),
                (2, 64, 64, 96),
                (2, 32, 32, 192),
                (2, 16, 16, 384),
                (2, 8, 8, 768),
            ),
            expected_output_dtype=("float32",) * 5,
        )

    def test_small(self):
        self.run_layer_test(
            Backbone,
            init_kwargs={"scales": None, "policy": "softswin_small-none"},
            input_shape=(2, 256, 256, 3),
            input_dtype="uint8",
            expected_output_shape=(
                (2, 128, 128, 48),
                (2, 64, 64, 96),
                (2, 32, 32, 192),
                (2, 16, 16, 384),
                (2, 8, 8, 768),
            ),
            expected_output_dtype=("float32",) * 5,
        )

    def test_base(self):
        self.run_layer_test(
            Backbone,
            init_kwargs={"scales": None, "policy": "softswin_base-none"},
            input_shape=(2, 384, 384, 3),
            input_dtype="uint8",
            expected_output_shape=(
                (2, 192, 192, 64),
                (2, 96, 96, 128),
                (2, 48, 48, 256),
                (2, 24, 24, 512),
                (2, 12, 12, 1024),
            ),
            expected_output_dtype=("float32",) * 5,
        )

    def test_large(self):
        self.run_layer_test(
            Backbone,
            init_kwargs={"scales": None, "policy": "softswin_large-none"},
            input_shape=(2, 384, 384, 3),
            input_dtype="uint8",
            expected_output_shape=(
                (2, 192, 192, 96),
                (2, 96, 96, 192),
                (2, 48, 48, 384),
                (2, 24, 24, 768),
                (2, 12, 12, 1536),
            ),
            expected_output_dtype=("float32",) * 5,
        )
