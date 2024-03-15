import numpy as np
import tensorflow as tf
from tf_keras import initializers, layers
from tf_keras.saving import register_keras_serializable
from tf_keras.src.utils.tf_utils import shape_type_conversion
from segme.common.convnormact import Conv
from segme.common.shape import get_shape


@register_keras_serializable(package='SegMe>Common')
class ChannelAttention(layers.Layer):
    def __init__(self, num_heads, qkv_bias=True, proj_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.proj_bias = proj_bias

    @shape_type_conversion
    def build(self, input_shape):
        self.channels = input_shape[-1]
        if self.channels is None:
            raise ValueError('Channel dimensions of the inputs should be defined. Found `None`.')

        self.qkv = Conv(self.channels * 3, 1, use_bias=False, name='qkv')
        if self.qkv_bias:
            self.q_bias = self.add_weight('q_bias', shape=[self.channels], initializer='zeros')
            self.v_bias = self.add_weight('v_bias', shape=[self.channels], initializer='zeros')

        self.scale = self.add_weight(
            'scale', shape=[self.num_heads, 1, 1],
            initializer=initializers.constant(np.log(10., dtype=self.dtype)),
            constraint=lambda s: tf.minimum(s, np.log(100., dtype=self.dtype)))

        self.proj = Conv(self.channels, 1, use_bias=self.proj_bias, name='proj')

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        qkv = self.qkv(inputs)
        if self.qkv_bias:
            k_bias = tf.zeros([self.channels], dtype=self.compute_dtype)
            qkv_bias = tf.concat([self.q_bias, k_bias, self.v_bias], axis=0)
            qkv = tf.nn.bias_add(qkv, qkv_bias)

        (batch, height, width), _ = get_shape(qkv, axis=[0, 1, 2])
        if 1 == self.num_heads:
            qkv = tf.reshape(qkv, [batch, 1, height * width, 3, self.channels])
        else:
            qkv = tf.reshape(qkv, [batch, height * width, self.num_heads, 3, self.channels // self.num_heads])
            qkv = tf.transpose(qkv, [0, 2, 1, 3, 4])
        q, k, v = tf.unstack(qkv, 3, axis=-2)

        q = tf.math.l2_normalize(q, axis=-2, epsilon=1.55e-5)
        k = tf.math.l2_normalize(k, axis=-2, epsilon=1.55e-5)

        attn = tf.matmul(q * tf.exp(self.scale), k, transpose_a=True)
        attn = tf.nn.softmax(attn)

        outputs = tf.transpose(tf.matmul(attn, v, transpose_b=True), perm=[0, 3, 1, 2])
        outputs = tf.reshape(outputs, [batch, height, width, self.channels])

        outputs = self.proj(outputs)
        outputs.set_shape(inputs.shape)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()

        config.update({
            'num_heads': self.num_heads,
            'qkv_bias': self.qkv_bias,
            'proj_bias': self.proj_bias
        })

        return config
