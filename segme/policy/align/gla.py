import numpy as np
from keras.src import initializers
from keras.src import layers
from keras.src import ops
from keras.src.layers.input_spec import InputSpec
from keras.src.saving import register_keras_serializable

from segme.common.attn.mincon import MinConstraint
from segme.common.attn.relbias import RelativeBias
from segme.common.attn.slide import DeformableConstraint
from segme.common.convnormact import Conv
from segme.common.convnormact import ConvAct
from segme.common.convnormact import Norm
from segme.common.resize import BilinearInterpolation
from segme.common.resize import NearestInterpolation
from segme.common.sequence import Sequence


@register_keras_serializable(package="SegMe>Policy>Align>GLA")
class GlaFeatureAlignment(layers.Layer):
    def __init__(
        self,
        filters,
        scale=2,
        fs_ratio=1.0,
        res_near=True,
        in_norm=False,
        in_mode="coarse",
        in_kernel=1,
        out_fine=False,
        out_kernel=1,
        out_norm=False,
        window_size=3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_spec = [
            InputSpec(ndim=4),  # fine
            InputSpec(ndim=4),  # coarse
        ]

        self.filters = filters
        self.scale = scale
        self.fs_ratio = fs_ratio
        self.res_near = res_near
        self.in_norm = in_norm
        self.in_mode = in_mode
        self.in_kernel = in_kernel
        self.out_fine = out_fine
        self.out_kernel = out_kernel
        self.out_norm = out_norm
        self.window_size = window_size

    def build(self, input_shape):
        channels = [shape[-1] for shape in input_shape]
        if None in channels:
            raise ValueError(
                "Channel dimension of the inputs should be defined. "
                "Found `None`."
            )
        self.input_spec = [
            InputSpec(ndim=4, axes={-1: channels[0]}),
            InputSpec(ndim=4, axes={-1: channels[1]}),
        ]

        if self.fs_ratio < 1.0:
            self.select_fine = SeFeatureSelection(
                self.fs_ratio, dtype=self.dtype_policy
            )
            self.select_fine.build(input_shape[0])
            self.select_coarse = SeFeatureSelection(
                self.fs_ratio, dtype=self.dtype_policy
            )
            self.select_coarse.build(input_shape[1])

            input_shape = (
                self.select_fine.compute_output_shape(input_shape[0]),
                self.select_coarse.compute_output_shape(input_shape[1]),
            )
            channels = [shape[-1] for shape in input_shape]

        if self.in_norm:
            self.norm_fine = Norm(dtype=self.dtype_policy)
            self.norm_fine.build(input_shape[0])

            self.norm_coarse = Norm(dtype=self.dtype_policy)
            self.norm_coarse.build(input_shape[1])

        if self.res_near:
            self.intnear = NearestInterpolation(
                self.scale, dtype=self.dtype_policy
            )
        else:
            self.intnear = BilinearInterpolation(
                self.scale, dtype=self.dtype_policy
            )

        if "l2h" == self.in_mode:
            self.gate_in = Conv(
                channels[1],
                self.in_kernel,
                activation="sigmoid",
                dtype=self.dtype_policy,
            )
            self.gate_in.build(input_shape[1][:-1] + (sum(channels),))
        elif "h2l" == self.in_mode:
            self.reduce_fine = Conv(
                channels[0], 3, strides=2, dtype=self.dtype_policy
            )
            self.reduce_fine.build(input_shape[0])

            self.gate_in = Conv(
                channels[1],
                self.in_kernel,
                activation="sigmoid",
                dtype=self.dtype_policy,
            )
            self.gate_in.build(input_shape[1][:-1] + (sum(channels),))
        else:  # just coarse
            self.gate_in = Conv(
                channels[1],
                self.in_kernel,
                activation="sigmoid",
                dtype=self.dtype_policy,
            )
            self.gate_in.build(input_shape[1])

        self.query_fine = Conv(channels[1], 1, dtype=self.dtype_policy)
        self.query_fine.build(input_shape[0])

        self.query_coarse = Conv(channels[1], 1, dtype=self.dtype_policy)
        self.query_coarse.build(input_shape[1])

        num_heads = max(channels[1] // 32, 1)
        self.attn = LocalAttention(
            self.scale,
            self.window_size,
            num_heads,
            qk_units=None,  # None - 16?
            cpb_units=num_heads * 8,
            dtype=self.dtype_policy,
        )
        self.attn.build([input_shape[0][:-1] + (channels[1],), input_shape[1]])

        if self.out_fine:
            self.gate_out = Conv(
                channels[1],
                self.in_kernel,
                activation="sigmoid",
                dtype=self.dtype_policy,
            )
            self.gate_out.build(input_shape[0][:-1] + (sum(channels),))
        else:
            self.gate_out = Conv(
                channels[1],
                self.in_kernel,
                activation="sigmoid",
                dtype=self.dtype_policy,
            )
            self.gate_out.build(input_shape[1])

        self.proj_fine = Conv(channels[1], 1, dtype=self.dtype_policy)
        self.proj_fine.build(input_shape[0])

        self.proj_out = Conv(self.filters, 1, dtype=self.dtype_policy)
        self.proj_out.build(input_shape[0][:-1] + (channels[1],))

        if self.out_norm:
            self.norm_out = Norm(dtype=self.dtype_policy)
            self.norm_out.build(input_shape[0][:-1] + (self.filters,))

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        fine, coarse = inputs

        if self.fs_ratio < 1.0:
            fine = self.select_fine(fine)
            coarse = self.select_coarse(coarse)

        if self.in_norm:
            fine = self.norm_fine(fine)
            coarse = self.norm_coarse(coarse)

        if "l2h" == self.in_mode:
            in_gate = ops.concatenate([fine, self.intnear(coarse)], axis=-1)
            in_gate = self.gate_in(in_gate)
            query = self.query_fine(fine) * in_gate + self.intnear(
                self.query_coarse(coarse)
            ) * (1.0 - in_gate)
        elif "h2l" == self.in_mode:
            fine2 = self.reduce_fine(fine)
            in_gate = ops.concatenate([fine2, coarse], axis=-1)
            in_gate = self.gate_in(in_gate)
            query = self.query_fine(fine) * self.intnear(
                in_gate
            ) + self.intnear(self.query_coarse(coarse) * (1.0 - in_gate))
        else:  # just coarse
            in_gate = self.gate_in(coarse)
            query = self.query_fine(fine) * self.intnear(
                in_gate
            ) + self.intnear(self.query_coarse(coarse) * (1.0 - in_gate))

        coarse = self.attn([query, coarse])

        if self.out_fine:
            out_gate = ops.concatenate([fine, coarse], axis=-1)
            out_gate = self.gate_out(out_gate)
        else:
            out_gate = self.gate_out(coarse)

        fine = self.proj_fine(fine)

        outputs = out_gate * fine + (1.0 - out_gate) * coarse
        outputs = self.proj_out(outputs)

        if self.out_norm:
            outputs = self.norm_out(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0][:-1] + (self.filters,)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "scale": self.scale,
                "fs_ratio": self.fs_ratio,
                "res_near": self.res_near,
                "in_norm": self.in_norm,
                "in_kernel": self.in_kernel,
                "out_fine": self.out_fine,
                "out_kernel": self.out_kernel,
                "in_mode": self.in_mode,
                "out_norm": self.out_norm,
                "window_size": self.window_size,
            }
        )

        return config


@register_keras_serializable(package="SegMe>Policy>Align>GLA")
class SeFeatureSelection(layers.Layer):
    def __init__(self, ratio, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=4)

        self.ratio = ratio

    def build(self, input_shape):
        channels = input_shape[-1]
        if channels is None:
            raise ValueError(
                "Channel dimension of the inputs should be defined. "
                "Found `None`."
            )
        self.input_spec = InputSpec(ndim=4, axes={-1: channels})

        filters = round(channels * self.ratio)

        self.se = Sequence(
            [
                layers.GlobalAveragePooling2D(
                    keepdims=True, dtype=self.dtype_policy
                ),
                ConvAct(
                    filters,
                    1,
                    kernel_initializer="variance_scaling",
                    dtype=self.dtype_policy,
                ),
                layers.Conv2D(
                    channels,
                    1,
                    activation="sigmoid",
                    kernel_initializer="variance_scaling",
                    dtype=self.dtype_policy,
                ),
            ]
        )
        self.se.build(input_shape)

        self.proj = Conv(filters, 1, dtype=self.dtype_policy)
        self.proj.build(input_shape)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        attention = self.se(inputs)

        # same as inputs * attention + inputs = SE + skip connection
        outputs = inputs * (attention + 1.0)

        outputs = self.proj(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (round(input_shape[-1] * self.ratio),)

    def get_config(self):
        config = super().get_config()
        config.update({"ratio": self.ratio})

        return config


@register_keras_serializable(package="SegMe>Policy>Align>GLA")
class LocalAttention(layers.Layer):
    def __init__(
        self,
        scale,
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
        self.input_spec = [
            InputSpec(ndim=4),  # query
            InputSpec(ndim=4),  # key, value
        ]

        self.scale = scale
        self.window_size = window_size
        self.num_heads = num_heads
        self.qk_units = qk_units
        self.qkv_bias = qkv_bias
        self.cpb_units = cpb_units
        self.dilation_rate = dilation_rate
        self.proj_bias = proj_bias

    def build(self, input_shape):
        self.channels = [shape[-1] for shape in input_shape]
        if None in self.channels:
            raise ValueError(
                "Channel dimensions of the inputs should be defined. "
                "Found `None`."
            )
        if any(channels % self.num_heads for channels in self.channels):
            raise ValueError(
                "Channel dimensions of the inputs should be a multiple of "
                "the number of heads."
            )

        self.v_channels = self.channels[1]
        self.v_units = self.v_channels // self.num_heads
        self.qk_units = self.qk_units or self.v_units
        self.qk_channels = self.qk_units * self.num_heads
        if self.v_units % self.qk_units or self.qk_units > self.v_units:
            qk_allowed = [
                i for i in range(1, self.v_units + 1) if not self.v_units % i
            ]
            raise ValueError(
                f"Provided QK units value is not supported. Allowed values "
                f"are: {qk_allowed}."
            )

        self.q = Conv(
            self.qk_channels,
            1,
            use_bias=False,
            name="q",
            dtype=self.dtype_policy,
        )
        self.q.build(input_shape[0])

        self.kv = Conv(
            self.qk_channels + self.v_channels,
            1,
            use_bias=False,
            name="kv",
            dtype=self.dtype_policy,
        )
        self.kv.build(input_shape[1])

        if self.qkv_bias:
            self.q_bias = self.add_weight(
                name="q_bias", shape=[self.qk_channels], initializer="zeros"
            )
            self.v_bias = self.add_weight(
                name="v_bias", shape=[self.v_channels], initializer="zeros"
            )

        static_kernel = ops.zeros(
            (self.window_size, self.window_size, 1, self.window_size**2),
            dtype=self.compute_dtype,
        )
        self.static_kernel = DeformableConstraint(self.window_size)(
            static_kernel
        )

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

        self.log_scale = self.add_weight(
            name="scale",
            shape=[self.num_heads, 1, 1],
            initializer=initializers.Constant(np.log(10.0, dtype=self.dtype)),
            constraint=MinConstraint(np.log(100.0, dtype=self.dtype)),
        )

        self.rel_bias = RelativeBias(
            1,  # TODO: add scale
            self.window_size,
            self.window_size,
            self.num_heads,
            cpb_units=self.cpb_units,
            name="rel_bias",
            dtype=self.dtype_policy,
        )
        self.rel_bias.build(None)

        self.proj = Conv(
            self.v_channels,
            1,
            use_bias=self.proj_bias,
            name="proj",
            dtype=self.dtype_policy,
        )
        self.proj.build(input_shape[1])

        super().build(input_shape)

    def deformable_initializer(self, shape, dtype):
        weight = initializers.GlorotUniform()(shape, dtype)
        weight = DeformableConstraint(self.window_size)(weight)

        return weight

    def call(self, inputs, **kwargs):
        q, kv = inputs
        batch, height, width = ops.shape(kv)[:3]

        q = self.q(q)
        kv = self.kv(kv)
        k, v = ops.split(kv, [self.qk_channels], axis=-1)
        if self.qkv_bias:
            q = ops.add(q, self.q_bias)
            v = ops.add(v, self.v_bias)

        k = ops.depthwise_conv(
            k,
            self.deformable_kernel,
            padding="same",
            dilation_rate=self.dilation_rate,
        )

        v_kernel = self.deformable_kernel
        if self.v_channels != self.qk_channels:
            v_kernel = ops.repeat(
                v_kernel, self.v_channels // self.qk_channels, axis=2
            )
        v = ops.depthwise_conv(
            v,
            v_kernel,
            padding="same",
            dilation_rate=self.dilation_rate,
        )

        q = ops.reshape(
            q,
            [
                batch,
                height,
                self.scale,
                width,
                self.scale,
                self.qk_units,
                self.num_heads,
            ],
        )
        q = ops.transpose(q, [0, 1, 3, 6, 2, 4, 5])
        q = ops.reshape(
            q,
            [
                batch,
                height,
                width,
                self.num_heads,
                self.scale**2,
                self.qk_units,
            ],
        )

        k = ops.reshape(
            k,
            [
                batch,
                height,
                width,
                self.qk_units,
                self.num_heads,
                self.window_size**2,
            ],
        )
        k = ops.transpose(k, [0, 1, 2, 4, 3, 5])

        v = ops.reshape(
            v,
            [
                batch,
                height,
                width,
                self.v_units,
                self.num_heads,
                self.window_size**2,
            ],
        )
        v = ops.transpose(v, [0, 1, 2, 4, 5, 3])

        q = ops.normalize(q, epsilon=3.94e-3)
        k = ops.normalize(k, epsilon=3.94e-3)

        attn = ops.matmul(q * ops.exp(self.log_scale), k)
        attn += self.attn_mask(height, width)
        attn = ops.softmax(attn)

        outputs = ops.matmul(attn, v)
        outputs = ops.reshape(
            outputs,
            [
                batch,
                height,
                width,
                self.num_heads,
                self.scale,
                self.scale,
                self.v_units,
            ],
        )
        outputs = ops.transpose(outputs, [0, 1, 4, 2, 5, 6, 3])
        outputs = ops.reshape(
            outputs,
            [batch, height * self.scale, width * self.scale, self.v_channels],
        )

        outputs = self.proj(outputs)

        return outputs

    def attn_mask(self, height, width):
        mask = ops.ones((1, height, width, 1), dtype=self.compute_dtype)
        mask = ops.depthwise_conv(
            mask,
            self.static_kernel,
            padding="same",
            dilation_rate=self.dilation_rate,
        )
        mask = ops.reshape(mask, [1, height, width, 1, 1, self.window_size**2])
        mask = -100.0 * ops.cast(mask == 0.0, self.compute_dtype)

        mask += self.rel_bias(None)[None]

        return mask

    def compute_output_shape(self, input_shape):
        return input_shape[0][:-1] + input_shape[1][-1:]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "scale": self.scale,
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
