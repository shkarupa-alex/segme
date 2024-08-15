import tensorflow as tf
from keras.src import testing

from segme.common.backbone import Backbone


class TestSwinV1(testing.TestCase):
    def test_tiny_224(self):
        self.run_layer_test(
            Backbone,
            init_kwargs={"scales": None, "policy": "swin_tiny_224-imagenet"},
            input_shape=(2, 224, 224, 3),
            input_dtype="int16",
            expected_output_shape=(
                (2, 56, 56, 96),
                (2, 28, 28, 192),
                (2, 14, 14, 384),
                (2, 7, 7, 768),
            ),
            expected_output_dtype=("float32",) * 4,
        )

    def test_small_224(self):
        self.run_layer_test(
            Backbone,
            init_kwargs={"scales": None, "policy": "swin_small_224-none"},
            input_shape=(2, 224, 224, 3),
            input_dtype="int16",
            expected_output_shape=(
                (2, 56, 56, 96),
                (2, 28, 28, 192),
                (2, 14, 14, 384),
                (2, 7, 7, 768),
            ),
            expected_output_dtype=("float32",) * 4,
        )

    def test_base_224(self):
        self.run_layer_test(
            Backbone,
            init_kwargs={"scales": None, "policy": "swin_base_224-none"},
            input_shape=(2, 224, 224, 3),
            input_dtype="int16",
            expected_output_shape=(
                (2, 56, 56, 128),
                (2, 28, 28, 256),
                (2, 14, 14, 512),
                (2, 7, 7, 1024),
            ),
            expected_output_dtype=("float32",) * 4,
        )

    def test_base_384(self):
        self.run_layer_test(
            Backbone,
            init_kwargs={"scales": None, "policy": "swin_base_384-none"},
            input_shape=(2, 384, 384, 3),
            input_dtype="int16",
            expected_output_shape=(
                (2, 96, 96, 128),
                (2, 48, 48, 256),
                (2, 24, 24, 512),
                (2, 12, 12, 1024),
            ),
            expected_output_dtype=("float32",) * 4,
        )

    def test_large_224(self):
        self.run_layer_test(
            Backbone,
            init_kwargs={"scales": None, "policy": "swin_large_224-none"},
            input_shape=(2, 224, 224, 3),
            input_dtype="int16",
            expected_output_shape=(
                (2, 56, 56, 192),
                (2, 28, 28, 384),
                (2, 14, 14, 768),
                (2, 7, 7, 1536),
            ),
            expected_output_dtype=("float32",) * 4,
        )

    def test_large_384(self):
        self.run_layer_test(
            Backbone,
            init_kwargs={"scales": None, "policy": "swin_large_384-none"},
            input_shape=(2, 384, 384, 3),
            input_dtype="int16",
            expected_output_shape=(
                (2, 96, 96, 192),
                (2, 48, 48, 384),
                (2, 24, 24, 768),
                (2, 12, 12, 1536),
            ),
            expected_output_dtype=("float32",) * 4,
        )


class TestSwinV2(testing.TestCase):
    def test_tiny_256(self):
        self.run_layer_test(
            Backbone,
            init_kwargs={"scales": None, "policy": "swin_v2_tiny_256-imagenet"},
            input_shape=(2, 256, 256, 3),
            input_dtype="int16",
            expected_output_shape=(
                (2, 64, 64, 96),
                (2, 32, 32, 192),
                (2, 16, 16, 384),
                (2, 8, 8, 768),
            ),
            expected_output_dtype=("float32",) * 4,
        )

    def test_small_256(self):
        self.run_layer_test(
            Backbone,
            init_kwargs={"scales": None, "policy": "swin_v2_small_256-none"},
            input_shape=(2, 256, 256, 3),
            input_dtype="int16",
            expected_output_shape=(
                (2, 64, 64, 96),
                (2, 32, 32, 192),
                (2, 16, 16, 384),
                (2, 8, 8, 768),
            ),
            expected_output_dtype=("float32",) * 4,
        )

    def test_base_256(self):
        self.run_layer_test(
            Backbone,
            init_kwargs={"scales": None, "policy": "swin_v2_base_256-none"},
            input_shape=(2, 256, 256, 3),
            input_dtype="int16",
            expected_output_shape=(
                (2, 64, 64, 128),
                (2, 32, 32, 256),
                (2, 16, 16, 512),
                (2, 8, 8, 1024),
            ),
            expected_output_dtype=("float32",) * 4,
        )

    def test_base_384(self):
        self.run_layer_test(
            Backbone,
            init_kwargs={"scales": None, "policy": "swin_v2_base_384-none"},
            input_shape=(2, 384, 384, 3),
            input_dtype="int16",
            expected_output_shape=(
                (2, 96, 96, 128),
                (2, 48, 48, 256),
                (2, 24, 24, 512),
                (2, 12, 12, 1024),
            ),
            expected_output_dtype=("float32",) * 4,
        )

    def test_large_256(self):
        self.run_layer_test(
            Backbone,
            init_kwargs={"scales": None, "policy": "swin_v2_large_256-none"},
            input_shape=(2, 256, 256, 3),
            input_dtype="int16",
            expected_output_shape=(
                (2, 64, 64, 192),
                (2, 32, 32, 384),
                (2, 16, 16, 768),
                (2, 8, 8, 1536),
            ),
            expected_output_dtype=("float32",) * 4,
        )

    def test_large_384(self):
        self.run_layer_test(
            Backbone,
            init_kwargs={"scales": None, "policy": "swin_v2_large_384-none"},
            input_shape=(2, 384, 384, 3),
            input_dtype="int16",
            expected_output_shape=(
                (2, 96, 96, 192),
                (2, 48, 48, 384),
                (2, 24, 24, 768),
                (2, 12, 12, 1536),
            ),
            expected_output_dtype=("float32",) * 4,
        )
