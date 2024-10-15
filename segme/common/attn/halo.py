import numpy as np
from keras.src import initializers
from keras.src import layers
from keras.src import ops
from keras.src.layers.input_spec import InputSpec
from keras.src.saving import register_keras_serializable

from segme.common.attn.mincon import MinConstraint
from segme.common.attn.part import halo_partition
from segme.common.attn.part import halo_partition_fused
from segme.common.attn.part import partition_apply_fused
from segme.common.attn.part import partition_reverse_fused
from segme.common.attn.relbias import RelativeBias
from segme.common.convnormact import Conv
from segme.common.convnormact import ConvNorm
from segme.common.pad import with_divisible_pad


@register_keras_serializable(package="SegMe>Common")
class HaloAttention(layers.Layer):
    def __init__(
        self,
        current_window,
        pretrain_window,
        num_heads,
        qk_units=None,
        qkv_bias=True,
        cpb_units=512,
        dilation_rate=1,
        proj_bias=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=4)

        if current_window % 2 or pretrain_window % 2:
            raise ValueError("Window size must be even.")

        self.current_window = current_window
        self.pretrain_window = pretrain_window
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
        self.halo_window = self.current_window * 2

        self.qkv = Conv(
            self.qk_channels * 2 + self.channels,
            1,
            use_bias=False,
            name="qkv",
            dtype=self.dtype_policy,
        )
        self.qkv.build(input_shape)

        self.kv_dw = ConvNorm(
            None,
            3,
            strides=2,
            use_bias=False,
            name="kv_dw",
            dtype=self.dtype_policy,
        )  # From PVTv2
        self.kv_dw.build(input_shape[:-1] + (self.qk_channels + self.channels,))

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
            self.halo_window // 2,
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

        outputs = with_divisible_pad(
            self.qkv_part, qkv, self.current_window * self.dilation_rate
        )
        outputs = self.proj(outputs)

        return outputs

    def qkv_part(self, qkv, pad_size, pad_val):
        pad_height, pad_width = pad_size

        q, kv = ops.split(qkv, [self.qk_channels], axis=-1)
        kv = self.kv_dw(kv)

        if self.qkv_bias:
            q = ops.add(q, self.q_bias)

            k_bias = ops.zeros([self.qk_channels], dtype=self.compute_dtype)
            kv_bias = ops.concatenate([k_bias, self.v_bias], axis=0)
            kv = ops.add(kv, kv_bias)

        q = partition_apply_fused(
            q,
            pad_height,
            pad_width,
            "window_size",
            self.current_window,
            self.num_heads,
            self.dilation_rate,
            qkv_mult=1,
        )
        if self.qk_units == self.v_units:
            kv = halo_partition_fused(
                kv,
                pad_height // 2,
                pad_width // 2,
                self.current_window // 2,
                self.halo_window // 2,
                self.num_heads,
                self.dilation_rate,
            )
            k, v = ops.split(kv, [self.qk_units], axis=-1)
        else:
            k, v = ops.split(kv, [self.qk_channels], axis=-1)
            k = halo_partition_fused(
                k,
                pad_height // 2,
                pad_width // 2,
                self.current_window // 2,
                self.halo_window // 2,
                self.num_heads,
                self.dilation_rate,
                qkv_mult=1,
            )
            v = halo_partition_fused(
                v,
                pad_height // 2,
                pad_width // 2,
                self.current_window // 2,
                self.halo_window // 2,
                self.num_heads,
                self.dilation_rate,
                qkv_mult=1,
            )

        outputs = self.qkv_attn(q, k, v, pad_size=pad_size, pad_val=pad_val)
        outputs = partition_reverse_fused(
            outputs,
            pad_height,
            pad_width,
            "window_size",
            self.current_window,
            self.num_heads,
            self.dilation_rate,
        )

        return outputs

    def qkv_attn(self, q, k, v, pad_size, pad_val):
        q = ops.normalize(q, epsilon=3.94e-3)
        k = ops.normalize(k, epsilon=3.94e-3)

        attn = ops.matmul(q * ops.exp(self.scale), ops.moveaxis(k, -1, -2))
        attn += self.attn_mask(pad_size, pad_val)
        attn = ops.softmax(attn)

        outputs = ops.matmul(attn, v)

        return outputs

    def attn_mask(self, pad_size, pad_val):
        return self.rel_bias(None) + self.pad_mask(pad_size, pad_val)

    def pad_mask(self, pad_size, pad_val):
        pad_height, pad_width = pad_size
        src_height = pad_height - sum(pad_val[:2])
        src_width = pad_width - sum(pad_val[2:])

        mask = ops.ones((1, src_height, src_width, 1), dtype=self.compute_dtype)
        mask = ops.pad(mask, [(0, 0), pad_val[:2], pad_val[2:], (0, 0)])
        mask = -ops.max_pool(
            -mask, (2, 2), strides=2, padding="same"
        )  # min pooling
        mask = halo_partition(
            mask,
            pad_height // 2,
            pad_width // 2,
            self.current_window // 2,
            self.halo_window // 2,
            self.dilation_rate,
        )
        mask = ops.squeeze(mask == 0.0, axis=-1)[:, :, None, None]
        mask = -100.0 * ops.cast(mask, self.compute_dtype)

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
                "dilation_rate": self.dilation_rate,
                "proj_bias": self.proj_bias,
            }
        )

        return config
