import tensorflow as tf
from keras import backend, layers, models
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from tensorflow_addons.layers import SpectralNormalization
from .encoder import Encoder
from .decoder import Decoder


@register_keras_serializable(package='SegMe>HRRN')
class HRRN(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [
            layers.InputSpec(ndim=4, axes={-1: 3}, dtype='uint8'),  # image
            layers.InputSpec(ndim=4, axes={-1: 1}, dtype='uint8')  # trimap
        ]

    @shape_type_conversion
    def build(self, input_shape):
        self.encoder = Encoder()
        self.decoder = Decoder()

        encoder_shape = input_shape[0][:-1] + (6,)
        self.encoder.build(encoder_shape)

        short_channels = self.encoder.compute_output_shape(encoder_shape)
        short_channels = [shape[-1] for shape in short_channels[:-1]]
        short_channels = [max(channel // 2, 32) for channel in short_channels]  # TODO
        # assert [32, 32, 128, 256, 512] == short_channels, short_channels # TODO

        self.short = [
            models.Sequential([
                SpectralNormalization(layers.Conv2D(filters, 3, padding='same', use_bias=False)),
                layers.ReLU(),
                layers.BatchNormalization(),
                SpectralNormalization(layers.Conv2D(filters, 3, padding='same', use_bias=False)),
                layers.ReLU(),
                layers.BatchNormalization()])
            for filters in short_channels
        ]

        self.proj = layers.Conv2D(1, 3, padding='same')

        super().build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        if training is None:
            training = backend.learning_phase()

        images, trimaps = inputs
        trimaps = tf.one_hot(trimaps[..., 0] // 86, 3, dtype='uint8')

        combos = tf.concat([images, trimaps], axis=-1)
        combos = tf.cast(combos, self.compute_dtype)

        features = self.encoder(combos)
        shortcuts = [short(feat) for short, feat in zip(self.short, features[:-1])]

        outputs = self.decoder(shortcuts + features[-1:])

        outputs = self.proj(outputs)
        outputs = tf.cast(outputs, 'float32')
        outputs = tf.nn.tanh(outputs) / 2. + .5

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[0][:-1] + (1,)

    def compute_output_signature(self, input_signature):
        outptut_signature = super().compute_output_signature(input_signature)

        return tf.TensorSpec(dtype='float32', shape=outptut_signature.shape)

    # def get_config(self):
    #     config = super().get_config()
    #     config.update({'psp_sizes': self.psp_sizes})
    #
    #     return config


def build_hrrn():
    inputs = [
        layers.Input(name='image', shape=[None, None, 3], dtype='uint8'),
        layers.Input(name='trimap', shape=[None, None, 1], dtype='uint8')
    ]
    outputs = HRRN()(inputs)
    model = models.Model(inputs=inputs, outputs=outputs, name='hrrn')

    return model
