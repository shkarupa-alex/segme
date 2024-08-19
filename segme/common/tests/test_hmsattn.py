from keras.src import layers
from keras.src import testing

from segme.common.hmsattn import HierarchicalMultiScaleAttention


class LogitsWithGuidance(layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        self.conv1 = layers.Conv2D(4, 3, strides=2, padding="same")
        self.conv2 = layers.Conv2D(2, 3, strides=2, padding="same")

        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        features = self.conv1(inputs)
        outputs = self.conv2(features)

        return outputs, features

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (2,), input_shape[:-1] + (4,)


class TestHierarchicalMultiScaleAttention(testing.TestCase):

    def test_layer(self):
        self.run_layer_test(
            HierarchicalMultiScaleAttention,
            init_kwargs={
                "layer": LogitsWithGuidance(),
                "scales": ((0.5,), (0.25, 0.5, 2.0)),
                "filters": 256,
                "dropout": 0.0,
            },
            input_shape=(2, 128, 128, 3),
            input_dtype="float32",
            expected_output_shape=(2, 128, 128, 2),
            expected_output_dtype="float32",
            custom_objects={"LogitsWithGuidance": LogitsWithGuidance},
        )
        self.run_layer_test(
            HierarchicalMultiScaleAttention,
            init_kwargs={
                "layer": LogitsWithGuidance(),
                "scales": ((0.5,), (0.5, 2.0)),
                "filters": 256,
                "dropout": 0.0,
            },
            input_shape=(2, 128, 128, 3),
            input_dtype="float32",
            expected_output_shape=(2, 128, 128, 2),
            expected_output_dtype="float32",
            custom_objects={"LogitsWithGuidance": LogitsWithGuidance},
        )
