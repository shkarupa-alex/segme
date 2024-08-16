import tensorflow as tf
from keras.src import layers
from keras.src.layers.input_spec import InputSpec
from keras.src.saving import register_keras_serializable

from segme.common.convnormact import Conv
from segme.common.convnormact import Norm
from segme.common.shape import get_shape


@register_keras_serializable(package="SegMe>Common")
class SimPool(layers.Layer):
    """Proposed in: https://arxiv.org/abs/2309.06891"""

    def __init__(self, num_heads, qkv_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=4)

        self.num_heads = num_heads
        self.qkv_bias = qkv_bias

    def build(self, input_shape):
        self.channels = input_shape[-1]
        if self.channels is None:
            raise ValueError(
                "Channel dimension of the inputs should be defined. Found `None`."
            )

        self.kv_norm = Norm(name="kv_norm", dtype=self.dtype_policy)
        self.kv_norm.build(input_shape)

        self.q_proj = Conv(
            self.channels, 1, use_bias=self.qkv_bias, name="q_proj", dtype=self.dtype_policy
        )
        self.q_proj.build((input_shape[0], 1, 1, input_shape[-1]))

        self.k_proj = Conv(
            self.channels, 1, use_bias=self.qkv_bias, name="k_proj", dtype=self.dtype_policy
        )
        self.k_proj.build(input_shape)

        self.scale = (self.channels // self.num_heads) ** -0.5

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        (batch, height, width), _ = get_shape(inputs, axis=[0, 1, 2])
        length = height * width

        q = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
        q = self.q_proj(q)
        q = tf.reshape(
            q, [batch, 1, self.num_heads, self.channels // self.num_heads]
        )
        q = tf.transpose(q, [0, 2, 1, 3])

        k = v = self.kv_norm(inputs)
        k = self.k_proj(k)
        k = tf.reshape(
            k, [batch, length, self.num_heads, self.channels // self.num_heads]
        )
        k = tf.transpose(k, [0, 2, 3, 1])
        v = tf.reshape(
            v, [batch, length, self.num_heads, self.channels // self.num_heads]
        )
        v = tf.transpose(v, [0, 2, 1, 3])

        attn = tf.matmul(q * self.scale, k)
        attn = tf.nn.softmax(attn)

        outputs = tf.matmul(attn, v)
        outputs = tf.reshape(outputs, [batch, self.channels])

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[:1] + input_shape[-1:]

    def get_config(self):
        config = super().get_config()
        config.update({"num_heads": self.num_heads, "qkv_bias": self.qkv_bias})

        return config
