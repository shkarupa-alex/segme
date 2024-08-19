import numpy as np
from keras.src import backend
from keras.src import testing

from segme.common.attn.slide import DeformableConstraint
from segme.common.attn.slide import SlideAttention


class TestSlideAttention(testing.TestCase):
    def test_layer(self):
        self.run_layer_test(
            SlideAttention,
            init_kwargs={
                "window_size": 3,
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
            SlideAttention,
            init_kwargs={
                "window_size": 5,
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
            SlideAttention,
            init_kwargs={
                "window_size": 3,
                "num_heads": 4,
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
            SlideAttention,
            init_kwargs={
                "window_size": 3,
                "num_heads": 2,
                "qk_units": 1,
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
            SlideAttention,
            init_kwargs={
                "window_size": 3,
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
            SlideAttention,
            init_kwargs={
                "window_size": 3,
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
            SlideAttention,
            init_kwargs={
                "window_size": 3,
                "num_heads": 2,
                "qk_units": None,
                "qkv_bias": True,
                "cpb_units": 512,
                "dilation_rate": 2,
                "proj_bias": True,
            },
            input_shape=(2, 15, 17, 4),
            input_dtype="float32",
            expected_output_shape=(2, 15, 17, 4),
            expected_output_dtype="float32",
        )
        self.run_layer_test(
            SlideAttention,
            init_kwargs={
                "window_size": 3,
                "num_heads": 2,
                "qk_units": None,
                "qkv_bias": True,
                "cpb_units": 512,
                "dilation_rate": 1,
                "proj_bias": False,
            },
            input_shape=(2, 15, 17, 4),
            input_dtype="float32",
            expected_output_shape=(2, 15, 17, 4),
            expected_output_dtype="float32",
        )


class TestDeformableConstraint(testing.TestCase):
    def test_value(self):
        kernel = np.random.uniform(high=0.1, size=(3, 3, 2, 9)).astype(
            "float32"
        )
        result = DeformableConstraint(3)(kernel)
        result = backend.convert_to_numpy(result)
        result = result.round().astype("int32")

        self.assertTrue((result[:, :, 0] == result[:, :, 1]).all())
        self.assertTrue(
            (
                result[0, 0, 0]
                == np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], "int32")
            ).all()
        )
        self.assertTrue(
            (
                result[0, 1, 0]
                == np.array([0, 1, 0, 0, 0, 0, 0, 0, 0], "int32")
            ).all()
        )
        self.assertTrue(
            (
                result[1, 0, 0]
                == np.array([0, 0, 0, 1, 0, 0, 0, 0, 0], "int32")
            ).all()
        )
