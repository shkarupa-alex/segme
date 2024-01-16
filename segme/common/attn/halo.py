import numpy as np
import tensorflow as tf
from keras import initializers, layers
from keras.saving import register_keras_serializable
from keras.src.utils.tf_utils import shape_type_conversion
from segme.common.attn.relbias import RelativeBias
from segme.common.convnormact import ConvNorm, Conv
from segme.common.pad import with_divisible_pad
from segme.common.part import halo_partition, halo_partition_fused, partition_apply_fused, partition_reverse_fused


@register_keras_serializable(package='SegMe>Common')
class HaloAttention(layers.Layer):
    def __init__(
            self, current_window, pretrain_window, num_heads, qk_units=None, qkv_bias=True, cpb_units=512,
            dilation_rate=1, proj_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        if current_window % 2 or pretrain_window % 2:
            raise ValueError('Window size must be even.')

        self.current_window = current_window
        self.pretrain_window = pretrain_window
        self.num_heads = num_heads
        self.qk_units = qk_units
        self.qkv_bias = qkv_bias
        self.cpb_units = cpb_units
        self.dilation_rate = dilation_rate
        self.proj_bias = proj_bias

    @shape_type_conversion
    def build(self, input_shape):
        self.channels = input_shape[-1]
        if self.channels is None:
            raise ValueError('Channel dimensions of the inputs should be defined. Found `None`.')
        if self.channels % self.num_heads:
            raise ValueError('Channel dimensions of the inputs should be a multiple of the number of heads.')

        self.v_units = self.channels // self.num_heads
        self.qk_units = self.qk_units or self.v_units
        self.qk_channels = self.qk_units * self.num_heads
        self.halo_window = self.current_window * 2

        self.qkv = Conv(self.qk_channels * 2 + self.channels, 1, use_bias=False, name='qkv')
        self.kv_dw = ConvNorm(None, 3, strides=2, use_bias=False, name='qkv_dw')  # From PVTv2

        if self.qkv_bias:
            self.q_bias = self.add_weight('q_bias', shape=[self.qk_channels], initializer='zeros')
            self.v_bias = self.add_weight('v_bias', shape=[self.channels], initializer='zeros')

        self.scale = self.add_weight(
            'scale', shape=[self.num_heads, 1, 1],
            initializer=initializers.constant(np.log(10., dtype=self.dtype)),
            constraint=lambda s: tf.minimum(s, np.log(100., dtype=self.dtype)))

        self.rel_bias = RelativeBias(
            self.current_window, self.halo_window // 2, self.pretrain_window, self.num_heads, cpb_units=self.cpb_units,
            name='rel_bias')

        self.proj = Conv(self.channels, 1, use_bias=self.proj_bias, name='proj')

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        qkv = self.qkv(inputs)

        outputs = with_divisible_pad(self.qkv_part, qkv, self.current_window * self.dilation_rate)
        outputs = self.proj(outputs)

        return outputs

    def qkv_part(self, qkv, pad_size, pad_val):
        pad_height, pad_width = pad_size

        q, kv = tf.split(qkv, [self.qk_channels, self.qk_channels + self.channels], axis=-1)
        kv = self.kv_dw(kv)

        if self.qkv_bias:
            q = tf.nn.bias_add(q, self.q_bias)

            k_bias = tf.zeros([self.qk_channels], dtype=self.compute_dtype)
            kv_bias = tf.concat([k_bias, self.v_bias], axis=0)
            kv = tf.nn.bias_add(kv, kv_bias)

        q = partition_apply_fused(
            q, pad_height, pad_width, 'window_size', self.current_window, self.num_heads, self.dilation_rate,
            qkv_mult=1)
        if self.qk_units == self.v_units:
            kv = halo_partition_fused(
                kv, pad_height // 2, pad_width // 2, self.current_window // 2, self.halo_window // 2, self.num_heads,
                self.dilation_rate)
            k, v = tf.split(kv, [self.qk_units, self.v_units], axis=-1)
        else:
            k, v = tf.split(kv, [self.qk_channels, self.channels], axis=-1)
            k = halo_partition_fused(
                k, pad_height // 2, pad_width // 2, self.current_window // 2, self.halo_window // 2, self.num_heads,
                self.dilation_rate, qkv_mult=1)
            v = halo_partition_fused(
                v, pad_height // 2, pad_width // 2, self.current_window // 2, self.halo_window // 2, self.num_heads,
                self.dilation_rate, qkv_mult=1)

        outputs = self.qkv_attn(q, k, v, pad_size=pad_size, pad_val=pad_val)
        outputs = partition_reverse_fused(
            outputs, pad_height, pad_width, 'window_size', self.current_window, self.num_heads, self.dilation_rate)

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
        return self.rel_bias(None) + self.pad_mask(pad_size, pad_val)

    def pad_mask(self, pad_size, pad_val):
        pad_height, pad_width = pad_size
        src_height = pad_height - sum(pad_val[:2])
        src_width = pad_width - sum(pad_val[2:])

        mask = tf.ones((1, src_height, src_width, 1), dtype=self.compute_dtype)
        mask = tf.pad(mask, [(0, 0), pad_val[:2], pad_val[2:], (0, 0)])
        mask = -tf.nn.max_pool2d(-mask, ksize=2, strides=2, padding='SAME')  # min pooling
        mask = halo_partition(
            mask, pad_height // 2, pad_width // 2, self.current_window // 2, self.halo_window // 2, self.dilation_rate)
        mask = tf.squeeze(mask == 0., axis=-1)[:, :, None, None]
        mask = -100. * tf.cast(mask, self.compute_dtype)

        return mask

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()

        config.update({
            'current_window': self.current_window,
            'pretrain_window': self.pretrain_window,
            'num_heads': self.num_heads,
            'qk_units': self.qk_units,
            'qkv_bias': self.qkv_bias,
            'cpb_units': self.cpb_units,
            'dilation_rate': self.dilation_rate,
            'proj_bias': self.proj_bias
        })

        return config
