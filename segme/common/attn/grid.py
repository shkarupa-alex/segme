import numpy as np
import tensorflow as tf
from keras.src import initializers
from keras.src import layers
from keras.src import ops
from keras.src.layers.input_spec import InputSpec
from keras.src.saving import register_keras_serializable

from segme.common.attn.mincon import MinConstraint
from segme.common.attn.relbias import RelativeBias
from segme.common.convnormact import Conv
from segme.common.pad import with_divisible_pad
from segme.common.part import partition_apply
from segme.common.part import partition_apply_fused
from segme.common.part import partition_reverse_fused


@register_keras_serializable(package="SegMe>Common")
class GridAttention(layers.Layer):
    def __init__(
        self,
        current_window,
        pretrain_window,
        num_heads,
        qk_units=None,
        qkv_bias=True,
        cpb_units=512,
        proj_bias=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=4)

        self.current_window = current_window
        self.pretrain_window = pretrain_window
        self.num_heads = num_heads
        self.qk_units = qk_units
        self.qkv_bias = qkv_bias
        self.cpb_units = cpb_units
        self.proj_bias = proj_bias

    def build(self, input_shape):
        self.channels = input_shape[-1]
        if self.channels is None:
            raise ValueError(
                "Channel dimensions of the inputs should be defined. "
                "Found `None`."
            )
        if self.channels % self.num_heads:
            raise ValueError(
                "Channel dimensions of the inputs should be a multiple of "
                "the number of heads."
            )

        self.v_units = self.channels // self.num_heads
        self.qk_units = self.qk_units or self.v_units
        self.qk_channels = self.qk_units * self.num_heads

        self.qkv = Conv(
            self.qk_channels * 2 + self.channels,
            1,
            use_bias=False,
            name="qkv",
            dtype=self.dtype_policy,
        )
        self.qkv.build(input_shape)

        if self.qkv_bias:
            self.q_bias = self.add_weight(
                name="q_bias", shape=[self.qk_channels], initializer="zeros"
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

        self.rel_bias = RelativeBias(
            self.current_window,
            self.current_window,
            self.pretrain_window,
            self.num_heads,
            cpb_units=self.cpb_units,
            name="rel_bias",
            dtype=self.dtype_policy,
        )
        self.rel_bias.build(None)

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
            k_bias = tf.zeros([self.qk_channels], dtype=self.compute_dtype)
            qkv_bias = tf.concat([self.q_bias, k_bias, self.v_bias], axis=0)
            qkv = tf.nn.bias_add(qkv, qkv_bias)

        outputs = with_divisible_pad(
            lambda padded, pad_size, pad_val: self.qkv_part(
                padded, pad_size, pad_val
            ),
            qkv,
            self.current_window,
        )

        outputs = self.proj(outputs)

        return outputs

    def qkv_part(self, qkv, pad_size, pad_val):
        pad_height, pad_width = pad_size

        if self.qk_units == self.v_units:
            qkv = partition_apply_fused(
                qkv,
                pad_height,
                pad_width,
                "grid_size",
                self.current_window,
                self.num_heads,
            )
            q, k, v = tf.split(
                qkv, [self.qk_units, self.qk_units, self.v_units], axis=-1
            )
        else:
            q, k, v = tf.split(
                qkv,
                [self.qk_channels, self.qk_channels, self.channels],
                axis=-1,
            )
            q = partition_apply_fused(
                q,
                pad_height,
                pad_width,
                "grid_size",
                self.current_window,
                self.num_heads,
                qkv_mult=1,
            )
            k = partition_apply_fused(
                k,
                pad_height,
                pad_width,
                "grid_size",
                self.current_window,
                self.num_heads,
                qkv_mult=1,
            )
            v = partition_apply_fused(
                v,
                pad_height,
                pad_width,
                "grid_size",
                self.current_window,
                self.num_heads,
                qkv_mult=1,
            )

        outputs = self.qkv_attn(q, k, v, pad_size=pad_size, pad_val=pad_val)
        outputs = partition_reverse_fused(
            outputs,
            pad_height,
            pad_width,
            "grid_size",
            self.current_window,
            self.num_heads,
        )

        return outputs

    def qkv_attn(self, q, k, v, pad_size, pad_val):
        q = tf.math.l2_normalize(q, axis=-1, epsilon=1.55e-5)
        k = tf.math.l2_normalize(k, axis=-1, epsilon=1.55e-5)

        attn = tf.matmul(q * tf.exp(self.scale), k, transpose_b=True)
        attn += self.attn_mask(pad_size, pad_val)
        attn = tf.nn.softmax(attn)

        outputs = tf.matmul(attn, v)

        return outputs

    def attn_mask(self, pad_size, pad_val):
        mask = self.rel_bias(None)

        mask = ops.cond(
            sum(pad_val) > 0,
            lambda: mask + self.pad_mask(pad_size, pad_val),
            lambda: tf.identity(mask),
        )

        return mask

    def pad_mask(self, pad_size, pad_val):
        pad_height, pad_width = pad_size
        src_height = pad_height - sum(pad_val[:2])
        src_width = pad_width - sum(pad_val[2:])

        mask = tf.zeros((1, src_height, src_width, 1), dtype="int64")
        mask = tf.pad(
            mask,
            [(0, 0), pad_val[:2], pad_val[2:], (0, 0)],
            constant_values=-100,
        )
        mask = partition_apply(
            mask, pad_height, pad_width, "grid_size", self.current_window, 1
        )
        mask = tf.squeeze(mask, axis=-1)[:, :, None, None]
        mask = tf.cast(mask, self.compute_dtype)

        return mask

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "current_window": self.current_window,
                "pretrain_window": self.pretrain_window,
                "num_heads": self.num_heads,
                "qk_units": self.qk_units,
                "qkv_bias": self.qkv_bias,
                "cpb_units": self.cpb_units,
                "proj_bias": self.proj_bias,
            }
        )

        return config
