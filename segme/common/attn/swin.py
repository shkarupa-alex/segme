import numpy as np
from keras.src import initializers
from keras.src import layers
from keras.src import ops
from keras.src.layers.input_spec import InputSpec
from keras.src.saving import register_keras_serializable

from segme.common.attn.mincon import MinConstraint
from segme.common.attn.part import partition_apply
from segme.common.attn.part import partition_apply_fused
from segme.common.attn.part import partition_reverse_fused
from segme.common.attn.relbias import RelativeBias
from segme.common.convnormact import Conv
from segme.common.pad import with_divisible_pad
from segme.ops import l2_normalize


@register_keras_serializable(package="SegMe>Common")
class SwinAttention(layers.Layer):
    def __init__(
        self,
        current_window,
        pretrain_window,
        num_heads,
        shift_mode,
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
        self.shift_mode = shift_mode % 5
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

        self.shift_size = self.current_window // 2
        self.shift_dir = {
            0: None,
            1: [1, 1],
            2: [1, -1],
            3: [-1, -1],
            4: [-1, 1],
        }[self.shift_mode]

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
            k_bias = ops.zeros([self.qk_channels], dtype=self.compute_dtype)
            qkv_bias = ops.concatenate(
                [self.q_bias, k_bias, self.v_bias], axis=0
            )
            qkv = ops.add(qkv, qkv_bias)

        apply_shift, shift_size = False, np.array([0, 0])
        if self.shift_dir is not None:
            curr_size = ops.shape(inputs)[1:3]
            static_size = all(map(lambda x: isinstance(x, int), curr_size))
            if static_size:
                curr_size = np.array(curr_size)
                with_shift = curr_size > self.current_window
                apply_shift = with_shift.any()
                shift_size = (
                    self.shift_size
                    * with_shift.astype(curr_size.dtype)
                    * self.shift_dir
                )
            else:
                curr_size = ops.cast(curr_size, "int32")
                with_shift = curr_size > self.current_window
                apply_shift = ops.any(with_shift)
                shift_size = (
                    self.shift_size
                    * ops.cast(with_shift, curr_size.dtype)
                    * self.shift_dir
                )

        qkv = ops.cond(
            apply_shift,
            lambda: ops.roll(qkv, -shift_size, [1, 2]),
            lambda: qkv,
        )

        outputs = with_divisible_pad(
            lambda padded, pad_size, pad_val: self.qkv_part(
                padded, pad_size, pad_val, apply_shift, shift_size
            ),
            qkv,
            self.current_window,
        )

        outputs = ops.cond(
            apply_shift,
            lambda: ops.roll(outputs, shift_size, [1, 2]),
            lambda: outputs,
        )

        outputs = self.proj(outputs)

        return outputs

    def qkv_part(self, qkv, pad_size, pad_val, apply_shift, shift_size):
        pad_height, pad_width = pad_size

        if self.qk_units == self.v_units:
            qkv = partition_apply_fused(
                qkv,
                pad_height,
                pad_width,
                "window_size",
                self.current_window,
                self.num_heads,
            )
            q, k, v = ops.split(
                qkv, [self.qk_units, self.qk_units * 2], axis=-1
            )
        else:
            q, k, v = ops.split(
                qkv,
                [self.qk_channels, self.qk_channels * 2],
                axis=-1,
            )
            q = partition_apply_fused(
                q,
                pad_height,
                pad_width,
                "window_size",
                self.current_window,
                self.num_heads,
                qkv_mult=1,
            )
            k = partition_apply_fused(
                k,
                pad_height,
                pad_width,
                "window_size",
                self.current_window,
                self.num_heads,
                qkv_mult=1,
            )
            v = partition_apply_fused(
                v,
                pad_height,
                pad_width,
                "window_size",
                self.current_window,
                self.num_heads,
                qkv_mult=1,
            )

        outputs = self.qkv_attn(
            q,
            k,
            v,
            pad_size=pad_size,
            pad_val=pad_val,
            apply_shift=apply_shift,
            shift_size=shift_size,
        )
        outputs = partition_reverse_fused(
            outputs,
            pad_height,
            pad_width,
            "window_size",
            self.current_window,
            self.num_heads,
        )

        return outputs

    def qkv_attn(self, q, k, v, pad_size, pad_val, apply_shift, shift_size):
        q = l2_normalize(q, axis=-1, epsilon=1.55e-5)
        k = l2_normalize(k, axis=-1, epsilon=1.55e-5)

        attn = ops.matmul(q * ops.exp(self.scale), ops.moveaxis(k, -1, -2))
        attn += self.attn_mask(attn, pad_size, pad_val, apply_shift, shift_size)
        attn = ops.softmax(attn)

        outputs = ops.matmul(attn, v)

        return outputs

    def attn_mask(self, attention, pad_size, pad_val, apply_shift, shift_size):
        mask = self.rel_bias(None)

        windows = ops.shape(attention)[1]
        mask_ = ops.repeat(mask, windows, axis=1)

        mask = ops.cond(
            apply_shift,
            lambda: mask_ + self.shift_mask(pad_size, pad_val, shift_size),
            lambda: ops.cond(
                sum(pad_val) > 0,
                lambda: mask_ + self.pad_mask(pad_size, pad_val),
                lambda: mask,
            ),
        )

        return mask

    def pad_mask(self, pad_size, pad_val):
        pad_height, pad_width = pad_size
        src_height = pad_height - sum(pad_val[:2])
        src_width = pad_width - sum(pad_val[2:])

        mask = ops.zeros((1, src_height, src_width, 1), dtype="int64")
        mask = ops.pad(
            mask,
            [(0, 0), pad_val[:2], pad_val[2:], (0, 0)],
            constant_values=-100,
        )
        mask = partition_apply(
            mask, pad_height, pad_width, "window_size", self.current_window, 1
        )
        mask = ops.squeeze(mask, axis=-1)[:, :, None, None]
        mask = ops.cast(mask, self.compute_dtype)

        return mask

    def shift_mask(self, pad_size, pad_val, shift_size):
        pad_height, pad_width = pad_size
        src_height = pad_height - sum(pad_val[:2])
        src_width = pad_width - sum(pad_val[2:])

        height_shift, width_shift = ops.unstack(ops.abs(shift_size))
        shift_height = ops.cast(
            src_height > self.current_window, height_shift.dtype
        )
        shift_width = ops.cast(
            src_width > self.current_window, width_shift.dtype
        )

        height_repeats = (
            src_height + (pad_val[1] - self.current_window) * shift_height,
            (self.current_window - height_shift - pad_val[1]) * shift_height,
            height_shift * shift_height,
        )
        width_repeats = (
            src_width + (pad_val[3] - self.current_window) * shift_width,
            (self.current_window - width_shift - pad_val[3]) * shift_width,
            width_shift * shift_width,
        )

        if self.shift_dir is not None:
            height_repeats = height_repeats[:: self.shift_dir[0]]
            width_repeats = width_repeats[:: self.shift_dir[1]]

        height_repeats = pad_val[:1] + height_repeats + pad_val[1:2]
        width_repeats = pad_val[2:3] + width_repeats + pad_val[3:]

        mask = np.arange(9, dtype="int32").reshape((3, 3)) + 1
        mask = ops.pad(mask, [(1, 1), (1, 1)])[None, ..., None]
        mask = ops.repeat(mask, height_repeats, axis=1)
        mask = ops.repeat(mask, width_repeats, axis=2)
        mask = partition_apply(
            mask, pad_height, pad_width, "window_size", self.current_window, 1
        )
        mask = ops.squeeze(mask, axis=-1)
        mask = mask[..., None] - mask[..., None, :]
        mask = ops.where(mask == 0, 0.0, -100.0)
        mask = ops.cast(mask, self.compute_dtype)
        mask = mask[:, :, None]

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
                "shift_mode": self.shift_mode,
                "qk_units": self.qk_units,
                "qkv_bias": self.qkv_bias,
                "cpb_units": self.cpb_units,
                "proj_bias": self.proj_bias,
            }
        )

        return config
