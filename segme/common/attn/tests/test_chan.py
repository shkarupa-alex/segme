from keras.src import testing

from segme.common.attn.chan import ChannelAttention


class TestChannelAttention(testing.TestCase):
    def test_layer(self):
        self.run_layer_test(
            ChannelAttention,
            init_kwargs={"num_heads": 2, "qkv_bias": True, "proj_bias": True},
            input_shape=(2, 16, 16, 4),
            input_dtype="float32",
            expected_output_shape=(2, 16, 16, 4),
            expected_output_dtype="float32",
        )
        self.run_layer_test(
            ChannelAttention,
            init_kwargs={"num_heads": 2, "qkv_bias": False, "proj_bias": True},
            input_shape=(2, 16, 16, 4),
            input_dtype="float32",
            expected_output_shape=(2, 16, 16, 4),
            expected_output_dtype="float32",
        )
        self.run_layer_test(
            ChannelAttention,
            init_kwargs={"num_heads": 4, "qkv_bias": True, "proj_bias": False},
            input_shape=(2, 16, 16, 4),
            input_dtype="float32",
            expected_output_shape=(2, 16, 16, 4),
            expected_output_dtype="float32",
        )
