import tensorflow as tf
from keras import layers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from .sff import SelectiveFeatureFusion
from ...common import SameConv, ResizeByScale


@register_keras_serializable(package='SegMe>MINet')
class Decoder(layers.Layer):
    def __init__(self, standardized=False, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [layers.InputSpec(ndim=4) for _ in range(4)]

        self.standardized = standardized

    @shape_type_conversion
    def build(self, input_shape):
        self.channels = input_shape[0][-1]
        if self.channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')

        self.reduce32 = SameConv(self.channels, 1)
        self.reduce16 = SameConv(self.channels, 1)
        self.reduce8 = SameConv(self.channels, 1)

        self.upscale2 = ResizeByScale(2)

        self.fuse1632 = SelectiveFeatureFusion(standardized=self.standardized)
        self.fuse816 = SelectiveFeatureFusion(standardized=self.standardized)
        self.fuse48 = SelectiveFeatureFusion(standardized=self.standardized)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        feats4, feats8, feats16, feats32 = inputs

        outputs = self.reduce32(feats32)
        outputs = self.upscale2(outputs)

        outputs = self.fuse1632([self.reduce16(feats16), outputs])
        outputs = self.upscale2(outputs)

        outputs = self.fuse816([self.reduce8(feats8), outputs])
        outputs = self.upscale2(outputs)

        outputs = self.fuse48([feats4, outputs])

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super().get_config()
        config.update({'standardized': self.standardized})

        return config
