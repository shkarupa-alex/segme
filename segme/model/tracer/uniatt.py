import tensorflow as tf
from keras import layers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from tensorflow_probability.python.stats import percentile
from .chnatt import ChannelAttention
from ...common import SameConv


@register_keras_serializable(package='SegMe>Tracer')
class UnionAttention(layers.Layer):
    def __init__(self, confidence, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        self.confidence = confidence

    @shape_type_conversion
    def build(self, input_shape):
        self.chatt = ChannelAttention(self.confidence)
        self.bn = layers.BatchNormalization()
        self.qkv = SameConv(3, 1, use_bias=False)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        batch, height, width, _ = tf.unstack(tf.shape(inputs))

        channel_outputs, channel_masks = self.chatt(inputs)
        channel_outputs = self.bn(channel_outputs)

        channel_masks = channel_masks[:, 0, 0]
        threshold = percentile(channel_masks, self.confidence * 100, axis=-1, keepdims=True, interpolation='linear')
        channel_masks = tf.where(channel_masks > threshold, channel_masks, tf.zeros_like(channel_masks))
        channel_masks = channel_masks[:, None, None]

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
