import tensorflow as tf
from keras import Sequential, layers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from segme.common.convnormact import ConvNormAct, ConvAct
from segme.common.ppm import PyramidPooling
from segme.common.interrough import BilinearInterpolation
from segme.common.head import HeadProjection


@register_keras_serializable(package='SegMe>Model>FBAMatting')
class Decoder(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [
            layers.InputSpec(ndim=4),  # features x2
            layers.InputSpec(ndim=4),  # features x4
            layers.InputSpec(ndim=4),  # features x32
            layers.InputSpec(ndim=4, axes={-1: 3}),  # image (scaled)
            layers.InputSpec(ndim=4, axes={-1: 3}),  # image (normalized)
            layers.InputSpec(ndim=4, axes={-1: 2}),  # twomap (scaled)
        ]

    @shape_type_conversion
    def build(self, input_shape):
        self.interpolate = BilinearInterpolation(None)

        self.ppm = PyramidPooling(256)

        self.conv_up1 = ConvNormAct(256, 3)
        self.conv_up2 = ConvNormAct(256, 3)
        self.conv_up3 = ConvNormAct(64, 3)
        self.conv_up4 = Sequential([
            ConvAct(32, 3),
            ConvAct(16, 3),
            HeadProjection(7)
        ])

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        feats2, feats4, feats32, imscal, imnorm, twomap = inputs

        outputs = self.ppm(feats32)
        outputs = self.conv_up1(outputs)

        outputs = self.interpolate([outputs, feats4])
        outputs = tf.concat([outputs, feats4], axis=-1)
        outputs = self.conv_up2(outputs)

        outputs = self.interpolate([outputs, feats2])
        outputs = tf.concat([outputs, feats2], axis=-1)
        outputs = self.conv_up3(outputs)

        outputs = self.interpolate([outputs, imscal])
        outputs = tf.concat([outputs, imscal, imnorm, twomap], axis=-1)
        outputs = self.conv_up4(outputs)

        outputs = tf.cast(outputs, 'float32')

        alpha, fgbg = tf.split(outputs, [1, 6], axis=-1)
        alpha = tf.clip_by_value(alpha, 0., 1.)
        fgbg = tf.nn.sigmoid(fgbg)
        alfgbg = tf.concat([alpha, fgbg], axis=-1)

        return alfgbg

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[-1][:-1] + (7,)

    def compute_output_signature(self, input_signature):
        outptut_signature = super().compute_output_signature(input_signature)

        return tf.TensorSpec(dtype='float32', shape=outptut_signature.shape)
