import numpy as np
import tensorflow as tf
from keras.src import initializers
from keras.src import layers
from keras.src.layers.input_spec import InputSpec
from keras.src.saving import register_keras_serializable

from segme.common.attn.mincon import MinConstraint
from segme.common.convnormact import Conv
from segme.common.shape import get_shape


@register_keras_serializable(package="SegMe>Common")
class ChannelAttention(layers.Layer):
    def __init__(self, num_heads, qkv_bias=True, proj_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=4)

        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.proj_bias = proj_bias

    def build(self, input_shape):
        self.channels = input_shape[-1]
        if self.channels is None:
            raise ValueError(
                "Channel dimensions of the inputs should be defined. "
                "Found `None`."
            )

        self.qkv = Conv(
            self.channels * 3,
            1,
            use_bias=False,
            name="qkv",
            dtype=self.dtype_policy,
        )
        self.qkv.build(input_shape)

        if self.qkv_bias:
            self.q_bias = self.add_weight(
                name="q_bias", shape=[self.channels], initializer="zeros"
            )
            self.v_bias = self.add_weight(
                name="v_bias", shape=[self.channels], initializer="zeros"
            )

        self.scale = self.add_weight(
            name="scale",
            shape=[self.num_heads, 1, 1],
            initializer=initializers.Constant(np.log(10.0, dtype=self.dtype)),
            constraint=MinConstraint(np.log(100.0, dtype=self.dtype)),
        )

        self.proj = Conv(
            self.channels,
            1,
            use_bias=self.proj_bias,
            name="proj",
            dtype=self.dtype_policy,
        )
        self.proj.build(input_shape)

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
            qkv = tf.reshape(
                qkv,
                [
                    batch,
                    height * width,
                    self.num_heads,
                    3,
                    self.channels // self.num_heads,
                ],
            )
            qkv = tf.transpose(qkv, [0, 2, 1, 3, 4])
        q, k, v = tf.unstack(qkv, 3, axis=-2)

        q = tf.math.l2_normalize(q, axis=-2, epsilon=1.55e-5)
        k = tf.math.l2_normalize(k, axis=-2, epsilon=1.55e-5)

        attn = tf.matmul(q * tf.exp(self.scale), k, transpose_a=True)
        attn = tf.nn.softmax(attn)

        outputs = tf.transpose(
            tf.matmul(attn, v, transpose_b=True), perm=[0, 3, 1, 2]
        )
        outputs = tf.reshape(outputs, [batch, height, width, self.channels])

        outputs = self.proj(outputs)
        outputs.set_shape(inputs.shape)

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "num_heads": self.num_heads,
                "qkv_bias": self.qkv_bias,
                "proj_bias": self.proj_bias,
            }
        )

        return config
