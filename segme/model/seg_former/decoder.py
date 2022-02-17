import tensorflow as tf
from keras import layers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from ...common import ConvNormRelu, ResizeByScale


@register_keras_serializable(package='SegMe>MINet')
class Decoder(layers.Layer):
    def __init__(self, units, standardized=False, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [layers.InputSpec(ndim=4) for _ in range(4)]

        self.units = units
        self.standardized = standardized

    @shape_type_conversion
    def build(self, input_shape):
        self.channels = input_shape[0][-1]
        if self.channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')

        self.mlp4 = layers.Dense(self.units)
        self.mlp8 = layers.Dense(self.units)
        self.mlp16 = layers.Dense(self.units)
        self.mlp32 = layers.Dense(self.units)

        self.upscale2 = ResizeByScale(2)
        self.upscale4 = ResizeByScale(4)
        self.upscale8 = ResizeByScale(8)

        self.fuse = ConvNormRelu(self.units, 1, standardized=self.standardized)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        feats4, feats8, feats16, feats32 = inputs

        feats32 = self.mlp32(feats32)
        feats32 = self.upscale8(feats32)

        feats16 = self.mlp16(feats16)
        feats16 = self.upscale4(feats16)

        feats8 = self.mlp8(feats8)
        feats8 = self.upscale2(feats8)

        feats4 = self.mlp4(feats4)

        outputs = tf.concat([feats32, feats16, feats8, feats4], axis=-1)
        outputs = self.fuse(outputs)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[0][:-1] + (self.units,)

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'standardized': self.standardized
        })

        return config
