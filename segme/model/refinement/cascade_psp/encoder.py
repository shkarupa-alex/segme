from keras.src import layers
from keras.src.layers.input_spec import InputSpec
from keras.src.saving import register_keras_serializable

from segme.common.backbone import Backbone
from segme.policy.backbone.utils import patch_channels


@register_keras_serializable(package="SegMe>Model>Refinement>CascadePSP")
class Encoder(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=4, axes={-1: 6})

    def build(self, input_shape):
        inputs = layers.Input(
            name="inputs", shape=(None, None, 6), dtype=self.compute_dtype
        )
        backbone = Backbone(
            [2, 4, 32],
            inputs,
            "resnet_rs_50_s8-imagenet",
            dtype=self.compute_dtype,
        )
        backbone = patch_channels(backbone, [0.408] * 3, [0.492**2] * 3)
        self.backbone = backbone

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        return self.backbone(inputs)

    def compute_output_shape(self, input_shape):
        return self.backbone.compute_output_shape(input_shape)
