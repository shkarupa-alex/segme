import tensorflow as tf
from keras import layers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from .chnatt import ChannelAttention
from ...common import AtrousSeparableConv, ConvNormRelu, ToChannelFirst, ToChannelLast


@register_keras_serializable(package='SegMe>Tracer')
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

        self.nchw = ToChannelFirst()
        self.nhwc = ToChannelLast()

        self.chatt = ChannelAttention(self.confidence)

        # DWS + DWConv
        self.conv_in = AtrousSeparableConv(self.channels, 3, activation='selu')
        self.conv_mid0 = AtrousSeparableConv(self.channels // 4, 1, activation='selu')
        self.conv_mid1 = AtrousSeparableConv(self.channels // 4, 3, activation='selu')
        self.conv_mid2 = AtrousSeparableConv(self.channels // 4, 3, activation='selu', dilation_rate=3)
        self.conv_mid3 = AtrousSeparableConv(self.channels // 4, 3, activation='selu', dilation_rate=5)
        self.conv_out = ConvNormRelu(1, 1, activation='selu')

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        _, height, width, _ = tf.unstack(tf.cast(tf.shape(inputs), 'float32'))

        inputs_ = self.nchw(inputs)  # TODO: check

        rows = tf.range(height, dtype='float32')[:, None]
        cols = tf.range(width, dtype='float32')[None]
        distance_ = tf.sqrt((rows - height / 2) ** 2 + (cols - height / 2) ** 2)
        mask_ = tf.cast(distance_ >= self.radius, 'complex64')[None, None]

        freq_ = tf.signal.fft2d(tf.cast(inputs_, 'complex64'))
        freq_ = tf.signal.fftshift(freq_)
        high_freq_ = freq_ * mask_
        high_freq_ = tf.signal.ifftshift(high_freq_)
        edges_ = tf.signal.ifft2d(high_freq_)
        edges_ = tf.abs(edges_)

        edges = self.nhwc(edges_)
        edges, _ = self.chatt(edges)
        edges = self.conv_in(edges)

        edges += tf.concat([
            self.conv_mid0(edges), self.conv_mid1(edges), self.conv_mid2(edges), self.conv_mid3(edges)],
            axis=-1)
        edges = tf.nn.relu(self.conv_out(edges))

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
