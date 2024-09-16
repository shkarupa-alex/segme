from keras.src import testing

from segme.common.backbone import Backbone


class TestMobileNet(testing.TestCase):
    def test_small(self):
        self.run_layer_test(
            Backbone,
            init_kwargs={
                "scales": None,
                "policy": "mobilenet_v3_small-imagenet",
            },
            input_shape=(2, 224, 224, 3),
            input_dtype="uint8",
            expected_output_shape=(
                (2, 112, 112, 16),
                (2, 56, 56, 72),
                (2, 28, 28, 96),
                (2, 14, 14, 288),
                (2, 7, 7, 576),
            ),
            expected_output_dtype=("float32",) * 5,
        )

    def test_large(self):
        self.run_layer_test(
            Backbone,
            init_kwargs={"scales": None, "policy": "mobilenet_v3_large-none"},
            input_shape=(2, 224, 224, 3),
            input_dtype="uint8",
            expected_output_shape=(
                (2, 112, 112, 64),
                (2, 56, 56, 72),
                (2, 28, 28, 240),
                (2, 14, 14, 672),
                (2, 7, 7, 960),
            ),
            expected_output_dtype=("float32",) * 5,
        )
