from keras.src import testing

from segme.common.backbone import Backbone


class TestEfficientNet(testing.TestCase):
    def test_small(self):
        self.run_layer_test(
            Backbone,
            init_kwargs={
                "scales": None,
                "policy": "efficientnet_v2_small-imagenet",
            },
            input_shape=(2, 224, 224, 3),
            input_dtype="uint8",
            expected_output_shape=(
                (2, 112, 112, 24),
                (2, 56, 56, 48),
                (2, 28, 28, 64),
                (2, 14, 14, 160),
                (2, 7, 7, 1280),
            ),
            expected_output_dtype=("float32",) * 5,
        )

    def test_medium(self):
        self.run_layer_test(
            Backbone,
            init_kwargs={
                "scales": None,
                "policy": "efficientnet_v2_medium-none",
            },
            input_shape=(2, 224, 224, 3),
            input_dtype="uint8",
            expected_output_shape=(
                (2, 112, 112, 24),
                (2, 56, 56, 48),
                (2, 28, 28, 80),
                (2, 14, 14, 176),
                (2, 7, 7, 1280),
            ),
            expected_output_dtype=("float32",) * 5,
        )

    def test_large(self):
        self.run_layer_test(
            Backbone,
            init_kwargs={
                "scales": None,
                "policy": "efficientnet_v2_large-none",
            },
            input_shape=(2, 224, 224, 3),
            input_dtype="uint8",
            expected_output_shape=(
                (2, 112, 112, 32),
                (2, 56, 56, 64),
                (2, 28, 28, 96),
                (2, 14, 14, 224),
                (2, 7, 7, 1280),
            ),
            expected_output_dtype=("float32",) * 5,
        )
