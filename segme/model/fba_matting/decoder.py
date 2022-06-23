import tensorflow as tf
from keras import Sequential, layers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from ...common import ConvNormRelu, PyramidPooling, SameConv, resize_by_sample


@register_keras_serializable(package='SegMe>FBAMatting')
class Decoder(layers.Layer):
    def __init__(self, pool_scales, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [
            layers.InputSpec(ndim=4),  # features x2
            layers.InputSpec(ndim=4),  # features x4
            layers.InputSpec(ndim=4),  # features x32
            layers.InputSpec(ndim=4, axes={-1: 3}),  # image (scaled)
            layers.InputSpec(ndim=4, axes={-1: 3}),  # image (normalized)
            layers.InputSpec(ndim=4, axes={-1: 2}),  # twomap (scaled)
        ]

        self.pool_scales = pool_scales

    @shape_type_conversion
    def build(self, input_shape):
        self.ppm = PyramidPooling(256, self.pool_scales, activation='leaky_relu', standardized=True)

        self.conv_up1 = ConvNormRelu(256, 3, activation='leaky_relu', standardized=True)
        self.conv_up2 = ConvNormRelu(256, 3, activation='leaky_relu', standardized=True)
        self.conv_up3 = ConvNormRelu(64, 3, activation='leaky_relu', standardized=True)
        self.conv_up4 = Sequential([
            SameConv(32, 3, activation='leaky_relu'),
            SameConv(16, 3, activation='leaky_relu'),
            SameConv(7, 1)
        ])

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        feats2, feats4, feats32, imscal, imnorm, twomap = inputs

        ppm_out = self.ppm(feats32)

        outputs = self.conv_up1(ppm_out)

        outputs = resize_by_sample([outputs, feats4])
        outputs = tf.concat([outputs, feats4], axis=-1)
        outputs = self.conv_up2(outputs)

        outputs = resize_by_sample([outputs, feats2])
        outputs = tf.concat([outputs, feats2], axis=-1)
        outputs = self.conv_up3(outputs)

        outputs = resize_by_sample([outputs, imscal])
        outputs = tf.concat([outputs, imscal, imnorm, twomap], axis=-1)
        outputs = self.conv_up4(outputs)
        outputs = tf.cast(outputs, 'float32')

        # TODO: original implementation
        # alpha = tf.clip_by_value(outputs[..., :1], 0., 1.)
        # fgbg = tf.nn.sigmoid(outputs[..., 1:])
        # alfgbg = tf.concat([alpha, fgbg], axis=-1)

        alfgbg = tf.nn.sigmoid(outputs)

        return alfgbg

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[-1][:-1] + (7,)

    def compute_output_signature(self, input_signature):
        outptut_signature = super().compute_output_signature(input_signature)

        return tf.TensorSpec(dtype='float32', shape=outptut_signature.shape)

    def get_config(self):
        config = super().get_config()
        config.update({'pool_scales': self.pool_scales})

        return config
