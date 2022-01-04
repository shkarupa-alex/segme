import tensorflow as tf
from keras import Sequential, layers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from tensorflow_addons.layers import GroupNormalization
from ...common import AdaptiveAveragePooling, StandardizedConv2D, resize_by_sample


@register_keras_serializable(package='SegMe>FBAMatting')
class Decoder(layers.Layer):
    def __init__(self, pool_scales, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [
            layers.InputSpec(ndim=4),  # features x2
            layers.InputSpec(ndim=4),  # features x4
            layers.InputSpec(ndim=4),  # features x32
            layers.InputSpec(ndim=4, axes={-1: 3}),  # image
            layers.InputSpec(ndim=4, axes={-1: 2}),  # twomap
        ]

        self.pool_scales = pool_scales

    @shape_type_conversion
    def build(self, input_shape):
        self.ppm = [Sequential([
            AdaptiveAveragePooling(scale),
            StandardizedConv2D(256, 1, padding='same'),
            GroupNormalization(),
            layers.LeakyReLU()
        ]) for scale in self.pool_scales]

        self.conv_up1 = Sequential([
            StandardizedConv2D(256, 3, padding='same'),
            GroupNormalization(),
            layers.LeakyReLU(),
            StandardizedConv2D(256, 3, padding='same'),
            GroupNormalization(),
            layers.LeakyReLU()
        ])
        self.conv_up2 = Sequential([
            StandardizedConv2D(256, 3, padding='same'),
            GroupNormalization(),
            layers.LeakyReLU()
        ])
        self.conv_up3 = Sequential([
            StandardizedConv2D(64, 3, padding='same'),
            GroupNormalization(),
            layers.LeakyReLU()
        ])

        self.conv_up4 = Sequential([
            layers.Conv2D(32, 3, padding='same'),
            layers.LeakyReLU(),
            layers.Conv2D(16, 3, padding='same'),
            layers.LeakyReLU(),
            layers.Conv2D(7, 1, padding='same'),
            layers.Activation('linear', dtype='float32')
        ])

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        feats2, feats4, feats32, image, twomap = inputs

        ppm_out = [feats32]
        for pool_scale in self.ppm:
            ppm_out.append(resize_by_sample([pool_scale(feats32), feats32]))

        ppm_out = layers.concatenate(ppm_out, axis=-1)
        outputs = self.conv_up1(ppm_out)

        outputs = resize_by_sample([outputs, feats4])
        outputs = layers.concatenate([outputs, feats4], axis=-1)
        outputs = self.conv_up2(outputs)

        outputs = resize_by_sample([outputs, feats2])
        outputs = layers.concatenate([outputs, feats2], axis=-1)
        outputs = self.conv_up3(outputs)

        outputs = resize_by_sample([outputs, image])
        outputs = layers.concatenate([outputs, image, twomap], axis=-1)
        outputs = self.conv_up4(outputs)

        alpha = tf.clip_by_value(outputs[..., :1], 0., 1.)
        fgbg = tf.nn.sigmoid(outputs[..., 1:])

        alfgbg = layers.concatenate([alpha, fgbg], axis=-1, dtype='float32')

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
