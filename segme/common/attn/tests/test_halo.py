import tensorflow as tf
from keras.src import testing

from segme.common.attn.halo import HaloAttention


class TestHaloAttention(testing.TestCase):
    def test_layer(self):
        self.run_layer_test(
            HaloAttention,
            init_kwargs={
                "current_window": 4,
                "pretrain_window": 4,
                "num_heads": 2,
                "qk_units": None,
                "qkv_bias": True,
                "cpb_units": 512,
                "dilation_rate": 1,
                "proj_bias": True,
            },
            input_shape=(2, 15, 17, 4),
            input_dtype="float32",
            expected_output_shape=(2, 15, 17, 4),
            expected_output_dtype="float32",
        )
        self.run_layer_test(
            HaloAttention,
            init_kwargs={
                "current_window": 8,
                "pretrain_window": 4,
                "num_heads": 2,
                "qk_units": None,
                "qkv_bias": True,
                "cpb_units": 512,
                "dilation_rate": 1,
                "proj_bias": True,
            },
            input_shape=(2, 14, 18, 4),
            input_dtype="float32",
            expected_output_shape=(2, 14, 18, 4),
            expected_output_dtype="float32",
        )
        self.run_layer_test(
            HaloAttention,
            init_kwargs={
                "current_window": 4,
                "pretrain_window": 4,
                "num_heads": 4,
                "qk_units": None,
                "qkv_bias": True,
                "cpb_units": 512,
                "dilation_rate": 1,
                "proj_bias": True,
            },
            input_shape=(2, 16, 16, 4),
            input_dtype="float32",
            expected_output_shape=(2, 16, 16, 4),
            expected_output_dtype="float32",
        )
        self.run_layer_test(
            HaloAttention,
            init_kwargs={
                "current_window": 4,
                "pretrain_window": 4,
                "num_heads": 2,
                "qk_units": 4,
                "qkv_bias": True,
                "cpb_units": 512,
                "dilation_rate": 1,
                "proj_bias": True,
            },
            input_shape=(2, 15, 17, 4),
            input_dtype="float32",
            expected_output_shape=(2, 15, 17, 4),
            expected_output_dtype="float32",
        )
        self.run_layer_test(
            HaloAttention,
            init_kwargs={
                "current_window": 4,
                "pretrain_window": 4,
                "num_heads": 2,
                "qk_units": None,
                "qkv_bias": False,
                "cpb_units": 512,
                "dilation_rate": 1,
                "proj_bias": True,
            },
            input_shape=(2, 15, 17, 4),
            input_dtype="float32",
            expected_output_shape=(2, 15, 17, 4),
            expected_output_dtype="float32",
        )
        self.run_layer_test(
            HaloAttention,
            init_kwargs={
                "current_window": 4,
                "pretrain_window": 4,
                "num_heads": 2,
                "qk_units": None,
                "qkv_bias": True,
                "cpb_units": 384,
                "dilation_rate": 1,
                "proj_bias": True,
            },
            input_shape=(2, 15, 17, 4),
            input_dtype="float32",
            expected_output_shape=(2, 15, 17, 4),
            expected_output_dtype="float32",
        )
        self.run_layer_test(
            HaloAttention,
            init_kwargs={
                "current_window": 4,
                "pretrain_window": 4,
                "num_heads": 2,
                "qk_units": None,
                "qkv_bias": True,
                "cpb_units": 512,
                "dilation_rate": 2,
                "proj_bias": True,
            },
            input_shape=(2, 13, 19, 4),
            input_dtype="float32",
            expected_output_shape=(2, 13, 19, 4),
            expected_output_dtype="float32",
        )
        self.run_layer_test(
            HaloAttention,
            init_kwargs={
                "current_window": 8,
                "pretrain_window": 4,
                "num_heads": 2,
                "qk_units": None,
                "qkv_bias": True,
                "cpb_units": 512,
                "dilation_rate": 2,
                "proj_bias": False,
            },
            input_shape=(2, 16, 16, 4),
            input_dtype="float32",
            expected_output_shape=(2, 16, 16, 4),
            expected_output_dtype="float32",
        )
