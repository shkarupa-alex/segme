import tensorflow as tf
from tf_keras import layers
from tf_keras.saving import register_keras_serializable
from tf_keras.src.utils.tf_utils import shape_type_conversion
from tensorflow_probability.python.stats import percentile
from segme.common.convnormact import Conv, Norm
from segme.common.shape import get_shape
from segme.model.sod.tracer.chnatt import ChannelAttention


@register_keras_serializable(package='SegMe>Model>SOD>Tracer')
class UnionAttention(layers.Layer):
    def __init__(self, confidence, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        self.confidence = confidence

    @shape_type_conversion
    def build(self, input_shape):
        self.chatt = ChannelAttention(self.confidence)
        self.norm = Norm()
        self.qkv = Conv(3, 1, use_bias=False)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        (batch, height, width), _ = get_shape(inputs, axis=[0, 1, 2])

        channel_outputs, channel_masks = self.chatt(inputs)
        channel_outputs = self.norm(channel_outputs)

        threshold = percentile(channel_masks, self.confidence * 100, axis=-1, keepdims=True, interpolation='linear')
        channel_masks *= tf.cast(channel_masks > threshold, self.compute_dtype)
        channel_outputs *= channel_masks

        qkv = self.qkv(channel_outputs)
        q, k, v = tf.split(qkv, 3, axis=-1)
        q_ = tf.reshape(q, [batch, height * width, 1])
        k_ = tf.reshape(k, [batch, height * width, 1])
        v_ = tf.reshape(v, [batch, height * width, 1])

        qk = tf.matmul(q_, k_, transpose_b=True)
        score = tf.nn.softmax(qk)

        attention = tf.matmul(score, v_)
        attention = tf.reshape(attention, [batch, height, width, 1])

        outputs = v + attention

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (1,)

    def get_config(self):
        config = super().get_config()
        config.update({'confidence': self.confidence})

        return config
