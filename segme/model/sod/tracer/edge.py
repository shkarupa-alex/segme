import tensorflow as tf
from tf_keras import layers
from tf_keras.saving import register_keras_serializable
from tf_keras.src.utils.tf_utils import shape_type_conversion
from segme.common.convnormact import ConvNormAct
from segme.common.sequence import Sequence
from segme.common.shape import get_shape
from segme.model.sod.tracer.chnatt import ChannelAttention


@register_keras_serializable(package='SegMe>Model>SOD>Tracer')
class FrequencyEdge(layers.Layer):
    def __init__(self, radius, confidence, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        self.radius = radius
        self.confidence = confidence

    @shape_type_conversion
    def build(self, input_shape):
        self.channels = input_shape[-1]
        if self.channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')
        if not self.channels // 4:
            raise ValueError('Channel dimension should be greater then 8.')

        self.chatt = ChannelAttention(self.confidence)

        self.conv_in = Sequence([
            ConvNormAct(None, 3),
            ConvNormAct(self.channels, 1)
        ])
        self.conv_mid0 = Sequence([
            ConvNormAct(None, 1),
            ConvNormAct(self.channels // 4, 1)
        ])
        self.conv_mid1 = Sequence([
            ConvNormAct(None, 3),
            ConvNormAct(self.channels // 4, 1)
        ])
        self.conv_mid2 = Sequence([
            ConvNormAct(None, 3, dilation_rate=3),
            ConvNormAct(self.channels // 4, 1)
        ])
        self.conv_mid3 = Sequence([
            ConvNormAct(None, 3, dilation_rate=5),
            ConvNormAct(self.channels // 4, 1)
        ])
        self.conv_out = ConvNormAct(1, 1)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        (height, width), _ = get_shape(inputs, axis=[1, 2])

        rows = tf.range(height, dtype='float32')[:, None]
        cols = tf.range(width, dtype='float32')[None]
        half = tf.cast(tf.minimum(height, width), 'float32') / 2.

        distance_ = tf.sqrt((rows - half) ** 2 + (cols - half) ** 2)
        mask_ = tf.cast(distance_ >= self.radius, 'complex64')[None, None]

        inputs_ = tf.transpose(inputs, [0, 3, 1, 2])
        inputs_ = tf.cast(inputs_, 'complex64')

        freq_ = tf.signal.fft2d(inputs_)
        freq_ = tf.signal.fftshift(freq_)
        high_freq_ = freq_ * mask_
        high_freq_ = tf.signal.ifftshift(high_freq_)
        edges_ = tf.signal.ifft2d(high_freq_)

        edges_ = tf.cast(tf.abs(edges_), self.compute_dtype)
        edges = tf.transpose(edges_, [0, 2, 3, 1])

        edges, _ = self.chatt(edges)
        edges = self.conv_in(edges)
        edges += tf.concat([
            self.conv_mid0(edges), self.conv_mid1(edges), self.conv_mid2(edges), self.conv_mid3(edges)], axis=-1)
        edges = self.conv_out(edges)
        edges = tf.nn.relu(edges)

        outputs = inputs + edges

        return outputs, edges

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape, input_shape[:-1] + (1,)

    def get_config(self):
        config = super().get_config()
        config.update({
            'radius': self.radius,
            'confidence': self.confidence
        })

        return config


def extract_edges(labels):
    mask = tf.cast(labels, 'float32')

    gy = tf.concat([
        mask[:, 1:2] - mask[:, 0:1],
        (mask[:, 2:] - mask[:, :-2]) / 2.,
        mask[:, -1:] - mask[:, -2:-1]
    ], axis=1)
    gx = tf.concat([
        mask[:, :, 1:2] - mask[:, :, 0:1],
        (mask[:, :, 2:] - mask[:, :, :-2]) / 2.,
        mask[:, :, -1:] - mask[:, :, -2:-1]
    ], axis=2)

    edge = tf.abs(gy) + tf.abs(gx)
    edge = tf.cast(edge > 0, 'int32')

    return edge
