import tensorflow as tf
from keras import layers
from keras.saving import register_keras_serializable
from keras.src.utils.tf_utils import shape_type_conversion
from segme.common.convnormact import Conv, Norm
from segme.common.sequence import Sequence


@register_keras_serializable(package='SegMe>Model>SOD>Tracer')
class ChannelAttention(layers.Layer):
    def __init__(self, confidence, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        self.confidence = confidence

    @shape_type_conversion
    def build(self, input_shape):
        self.channels = input_shape[-1]
        if self.channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')

        self.avg = Sequence([
            layers.GlobalAvgPool2D(keepdims=True),
            Norm(),
            layers.Dropout(self.confidence)
        ])
        self.qkv = Conv(self.channels * 3, 1, use_bias=False)
        self.proj = Conv(self.channels, 1, use_bias=False, activation='sigmoid')

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        channels = self.avg(inputs)

        qkv = self.qkv(channels)
        q, k, v = tf.split(qkv, 3, axis=-1)

        qk = tf.matmul(q, k, transpose_a=True)
        score = tf.nn.softmax(qk)

        attention = tf.matmul(v, score)
        attention = self.proj(attention)

        output = inputs * (attention + 1.)

        return output, attention

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape, input_shape[:1] + (1, 1, self.channels)

    def get_config(self):
        config = super().get_config()
        config.update({'confidence': self.confidence})

        return config
