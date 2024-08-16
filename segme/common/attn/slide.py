import numpy as np
import tensorflow as tf
from keras.src import initializers
from keras.src import layers, constraints, backend
from keras.src.layers.input_spec import InputSpec
from keras.src.saving import register_keras_serializable

from segme.common.attn.relbias import RelativeBias
from segme.common.convnormact import Conv
from segme.common.shape import get_shape
from segme.common.attn.mincon import MinConstraint


@register_keras_serializable(package="SegMe>Common")
class SlideAttention(layers.Layer):
    def __init__(
        self,
        window_size,
        num_heads,
        qk_units=None,
        qkv_bias=True,
        cpb_units=512,
        dilation_rate=1,
        proj_bias=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=4)

        self.window_size = window_size
        self.num_heads = num_heads
        self.qk_units = qk_units
        self.qkv_bias = qkv_bias
        self.cpb_units = cpb_units
        self.dilation_rate = dilation_rate
        self.proj_bias = proj_bias

    def build(self, input_shape):
        self.channels = input_shape[-1]
        if self.channels is None:
            raise ValueError(
                "Channel dimensions of the inputs should be defined. Found `None`."
            )
        if self.channels % self.num_heads:
            raise ValueError(
                "Channel dimensions of the inputs should be a multiple of the number of heads."
            )

        self.v_units = self.channels // self.num_heads
        self.qk_units = self.qk_units or self.v_units
        self.qk_channels = self.qk_units * self.num_heads
        if self.v_units % self.qk_units or self.qk_units > self.v_units:
            qk_allowed = [
                i for i in range(1, self.v_units + 1) if not self.v_units % i
            ]
            raise ValueError(
                f"Provided QK units value is not supported. Allowed values are: {qk_allowed}."
            )

        self.qkv = Conv(
            self.qk_channels * 2 + self.channels, 1, use_bias=False, name="qkv", dtype=self.dtype_policy
        )
        self.qkv.build(input_shape)

        if self.qkv_bias:
            self.q_bias = self.add_weight(
                name="q_bias", shape=[self.qk_channels], initializer="zeros"
            )
            self.v_bias = self.add_weight(
                name="v_bias", shape=[self.channels], initializer="zeros"
            )

        static_kernel = np.zeros(
            (self.window_size, self.window_size, 1, self.window_size**2),
            dtype="bool",
        )
        for i in range(self.window_size**2):
            static_kernel[i // self.window_size, i % self.window_size, :, i] = (
                True
            )
        self.static_mask = tf.cast(
            static_kernel.repeat(self.qk_channels, axis=2), "bool"
        )
        self.static_kernel = tf.cast(static_kernel, self.compute_dtype)

        self.deformable_kernel = self.add_weight(
            shape=(
                self.window_size,
                self.window_size,
                self.qk_channels,
                self.window_size**2,
            ),
            initializer=self.deformable_initializer,
            constraint=DeformableConstraint(self.window_size),
            name="deformable_kernel",
        )

        self.scale = self.add_weight(
            name="scale",
            shape=[self.num_heads, 1, 1],
            initializer=initializers.Constant(np.log(10.0, dtype=self.dtype)),
            constraint=MinConstraint(np.log(100.0, dtype=self.dtype)),
        )

        self.rel_bias = RelativeBias(
            1,
            self.window_size,
            self.window_size,
            self.num_heads,
            cpb_units=self.cpb_units,
            name="rel_bias", dtype=self.dtype_policy
        )
        self.rel_bias.build(None)

        self.proj = Conv(self.channels, 1, use_bias=self.proj_bias, name="proj", dtype=self.dtype_policy)
        self.proj.build(input_shape)

        super().build(input_shape)

    def deformable_initializer(self, shape, dtype):
        weight = initializers.GlorotUniform()(shape, dtype)
        weight += tf.cast(self.static_kernel, dtype)
        weight = DeformableConstraint(self.window_size)(weight)

        return weight

    def call(self, inputs, **kwargs):
        qkv = self.qkv(inputs)
        if self.qkv_bias:
            k_bias = tf.zeros([self.qk_channels], dtype=self.compute_dtype)
            qkv_bias = tf.concat([self.q_bias, k_bias, self.v_bias], axis=0)
            qkv = tf.nn.bias_add(qkv, qkv_bias)

        q, k, v = tf.split(
            qkv, [self.qk_channels, self.qk_channels, self.channels], axis=-1
        )

        k = tf.nn.depthwise_conv2d(
            k,
            self.deformable_kernel,
            strides=[1] * 4,
            padding="SAME",
            dilations=[self.dilation_rate, self.dilation_rate],
        )

        v_kernel = self.deformable_kernel
        if self.channels != self.qk_channels:
            v_kernel = tf.repeat(
                v_kernel, self.channels // self.qk_channels, axis=2
            )
        v = tf.nn.depthwise_conv2d(
            v,
            v_kernel,
            strides=[1] * 4,
            padding="SAME",
            dilations=[self.dilation_rate, self.dilation_rate],
        )

        (batch, height, width), _ = get_shape(inputs, axis=[0, 1, 2])
        q = tf.reshape(
            q, [batch, height, width, self.num_heads, 1, self.qk_units]
        )
        k = tf.reshape(
            k,
            [
                batch,
                height,
                width,
                self.num_heads,
                self.qk_units,
                self.window_size**2,
            ],
        )
        v = tf.reshape(
            v,
            [
                batch,
                height,
                width,
                self.num_heads,
                self.v_units,
                self.window_size**2,
            ],
        )

        q = tf.math.l2_normalize(q, axis=-1, epsilon=1.55e-5)
        k = tf.math.l2_normalize(k, axis=-2, epsilon=1.55e-5)

        attn = tf.matmul(q * tf.exp(self.scale), k)
        attn += self.attn_mask(height, width)
        attn = tf.nn.softmax(attn)

        outputs = tf.matmul(attn, v, transpose_b=True)
        outputs = tf.reshape(outputs, [batch, height, width, self.channels])

        outputs = self.proj(outputs)

        return outputs

    def attn_mask(self, height, width):
        mask = tf.ones((1, height, width, 1), dtype=self.compute_dtype)
        mask = tf.nn.depthwise_conv2d(
            mask,
            self.static_kernel,
            strides=[1] * 4,
            padding="SAME",
            dilations=[self.dilation_rate, self.dilation_rate],
        )
        mask = tf.reshape(mask, [1, height, width, 1, 1, self.window_size**2])
        mask = -100.0 * tf.cast(mask == 0.0, self.compute_dtype)

        mask += self.rel_bias(None)[None]

        return mask

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "window_size": self.window_size,
                "num_heads": self.num_heads,
                "qk_units": self.qk_units,
                "qkv_bias": self.qkv_bias,
                "cpb_units": self.cpb_units,
                "dilation_rate": self.dilation_rate,
                "proj_bias": self.proj_bias,
            }
        )

        return config


@register_keras_serializable(package="SegMe>Common")
class DeformableConstraint(constraints.Constraint):
    def __init__(self, window_size):
        self.window_size = window_size

        static_mask = np.zeros(
            (self.window_size, self.window_size, 1, self.window_size**2),
            dtype="bool",
        )
        for i in range(self.window_size**2):
            static_mask[i // self.window_size, i % self.window_size, 0, i] = (
                True
            )
        self.static_mask = backend.convert_to_tensor(static_mask)

    def __call__(self, w):
        w = backend.convert_to_tensor(w)

        if 4 != len(w.shape) or self.window_size != self.static_mask.shape[0] or self.window_size != self.static_mask.shape[1] or self.window_size ** 2 != self.static_mask.shape[3]:
            raise ValueError(
                f"Expecting weight shape to be ({self.window_size}, {self.window_size}, *, {self.window_size ** 2}), got {w.shape}")

        return tf.where(self.static_mask, 1.0, w)

    def get_config(self):
        return {'window_size': self.window_size}
