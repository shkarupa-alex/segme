import numpy as np
import tensorflow as tf
from keras import initializers, layers
from keras.saving.object_registration import register_keras_serializable
from keras.utils.control_flow_util import smart_cond
from keras.utils.tf_utils import shape_type_conversion
from segme.common.convnormact import ConvNorm, Conv, Act
from segme.common.pad import with_divisible_pad
from segme.common.sequent import Sequential
from segme.policy.backbone.diy.coma.part import partition_apply, partition_apply_fused, partition_reverse_fused
from segme.policy.backbone.diy.coma.part import with_partition_fused, halo_partition, halo_partition_fused


@register_keras_serializable(package='SegMe>Policy>Backbone>DIY>CoMA')
class DHMSA(layers.Layer):
    def __init__(self, current_window, pretrain_window, num_heads, dilation_rate=1, use_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        if current_window < pretrain_window:
            raise ValueError('Actual window size should not be less then pretrain one.')

        if current_window % 2 or pretrain_window % 2:
            raise ValueError('Window size must be even.')

        self.current_window = current_window
        self.pretrain_window = pretrain_window
        self.num_heads = num_heads
        self.dilation_rate = dilation_rate
        self.use_bias = use_bias

    @shape_type_conversion
    def build(self, input_shape):
        self.channels = input_shape[-1]
        if self.channels is None:
            raise ValueError('Channel dimensions of the inputs should be defined. Found `None`.')

        self.window_length = self.current_window ** 2
        self.halo_window = self.current_window * 2
        self.halo_length = self.halo_window ** 2

        self.qkv = Conv(self.channels * 3, 1, use_bias=False, name='qkv')
        self.qkv_dw = ConvNorm(None, 3, use_bias=False, name='qkv_dw')  # From CvT

        if self.use_bias:
            self.q_bias = self.add_weight('q_bias', shape=[self.channels], initializer='zeros')
            self.v_bias = self.add_weight('v_bias', shape=[self.channels], initializer='zeros')

        self.scale = self.add_weight(
            'scale', shape=[self.num_heads, 1, 1],
            initializer=initializers.constant(np.log(10., dtype=self.dtype)),
            constraint=lambda s: tf.minimum(s, np.log(100., dtype=self.dtype)))

        self.cpb = Sequential([
            layers.Dense(512, name='cpb_fc0'),
            Act(name='cpb_act'),
            layers.Dense(self.num_heads, activation='sigmoid', use_bias=False, name='cpb_fc1')], name='cpb')

        self.proj = Conv(self.channels, 1, use_bias=False, name='proj')

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        qkv = self.qkv(inputs)
        qkv = self.qkv_dw(qkv)

        if self.use_bias:
            k_bias = tf.zeros([self.channels], dtype=self.compute_dtype)
            use_bias = tf.concat([self.q_bias, k_bias, self.v_bias], axis=0)
            qkv = tf.nn.bias_add(qkv, use_bias)

        outputs = with_divisible_pad(self.qkv_part, qkv, self.current_window * self.dilation_rate)

        outputs = self.proj(outputs)

        return outputs

    def qkv_part(self, qkv, pad_size, pad_val):
        _, pad_height, pad_width = pad_size

        q, kv = tf.split(qkv, [self.channels, self.channels * 2], axis=-1)
        q_part = partition_apply_fused(
            q, pad_height, pad_width, 'window_size', self.current_window, 1, self.num_heads, self.dilation_rate)
        kv_part = halo_partition_fused(
            kv, pad_height, pad_width, self.current_window, self.current_window * 2, 2, self.num_heads,
            self.dilation_rate)

        parted = self.qkv_attn(q_part, kv_part, pad_size=pad_size, pad_val=pad_val)
        parted = partition_reverse_fused(
            parted, pad_height, pad_width, 'window_size', self.current_window, self.num_heads, self.dilation_rate)

        return parted

    def qkv_attn(self, q, kv, pad_size, pad_val):
        q = tf.squeeze(q, axis=-2)
        k, v = tf.unstack(kv, 2, axis=-2)

        q = tf.math.l2_normalize(q, axis=-1, epsilon=1.55e-5)
        k = tf.math.l2_normalize(k, axis=-1, epsilon=1.55e-5)

        attn = tf.matmul(q * tf.exp(self.scale), k, transpose_b=True)
        attn = self.rel_bias(attn)
        attn = self.pad_mask(attn, pad_size, pad_val)
        attn = tf.nn.softmax(attn)

        outputs = tf.matmul(attn, v)

        return outputs

    def rel_bias(self, attn):
        reltab = np.arange(1 - self.current_window * 3 // 2, self.current_window * 3 // 2).astype('float32')
        reltab = np.stack(np.meshgrid(reltab, reltab, indexing='ij'))
        reltab = np.transpose(reltab, [1, 2, 0])[None]
        reltab *= 8. / (self.pretrain_window - 1.)
        reltab = np.sign(reltab) * np.log1p(np.abs(reltab)) / np.log(8)
        reltab = tf.cast(reltab, self.compute_dtype)

        relidx0 = np.arange(self.current_window)
        relidx0 = np.stack(np.meshgrid(relidx0, relidx0, indexing='ij'), axis=0)
        relidx0 = np.reshape(relidx0, [2, -1])
        relidx1 = np.arange(self.halo_window)
        relidx1 = np.stack(np.meshgrid(relidx1, relidx1, indexing='ij'), axis=0)
        relidx1 = np.reshape(relidx1, [2, -1])
        relidx = relidx0[:, :, None] - relidx1[:, None]
        relidx = relidx + (self.halo_window - 1)
        relidx = relidx[0] * (3 * self.current_window - 1) + relidx[1]
        relidx = np.reshape(relidx, [-1])
        relidx = tf.cast(relidx, 'int64')

        bias = self.cpb(reltab)
        bias = tf.reshape(bias, [-1, self.num_heads])
        bias = tf.gather(bias, relidx) * 16.
        bias = tf.reshape(bias, [self.window_length, self.halo_length, -1])
        bias = tf.transpose(bias, perm=[2, 0, 1])[None]

        attn += bias

        return attn

    def pad_mask(self, attn, pad_size, pad_val):
        batch_size, pad_height, pad_width = pad_size
        src_height, src_width = pad_height - pad_val[0], pad_width - pad_val[1]

        mask = tf.ones((1, src_height, src_width, 1), dtype='float32')
        mask = tf.pad(mask, [(0, 0), (0, pad_val[0]), (0, pad_val[1]), (0, 0)])
        mask = halo_partition(
            mask, pad_height, pad_width, self.current_window, self.current_window * 2, self.dilation_rate)
        mask = tf.squeeze(mask == 0., axis=-1)[None, :, None, None]
        mask = -100. * tf.cast(mask, self.compute_dtype)

        num_windows = pad_height * pad_width // self.window_length
        attn = tf.reshape(attn, shape=[batch_size, num_windows, self.num_heads, self.window_length, self.halo_length])
        attn += mask
        attn = tf.reshape(attn, shape=[batch_size * num_windows, self.num_heads, self.window_length, self.halo_length])

        return attn

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()

        config.update({
            'current_window': self.current_window,
            'pretrain_window': self.pretrain_window,
            'num_heads': self.num_heads,
            'dilation_rate': self.dilation_rate,
            'use_bias': self.use_bias
        })

        return config


@register_keras_serializable(package='SegMe>Policy>Backbone>DIY>CoMA')
class CHMSA(layers.Layer):
    def __init__(self, num_heads, use_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        self.num_heads = num_heads
        self.use_bias = use_bias

    @shape_type_conversion
    def build(self, input_shape):
        self.channels = input_shape[-1]
        if self.channels is None:
            raise ValueError('Channel dimensions of the inputs should be defined. Found `None`.')

        self.qkv = Conv(self.channels * 3, 1, use_bias=False, name='qkv')
        self.qkv_dw = ConvNorm(None, 3, use_bias=False, name='qkv_dw')  # From CvT

        if self.use_bias:
            self.q_bias = self.add_weight('q_bias', shape=[self.channels], initializer='zeros')
            self.v_bias = self.add_weight('v_bias', shape=[self.channels], initializer='zeros')

        self.scale = self.add_weight(
            'scale', shape=[self.num_heads, 1, 1],
            initializer=initializers.constant(np.log(10., dtype=self.dtype)),
            constraint=lambda s: tf.minimum(s, np.log(100., dtype=self.dtype)))

        self.proj = layers.Dense(self.channels, use_bias=False, name='proj')

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        batch, height, width, _ = tf.unstack(tf.shape(inputs))

        qkv = self.qkv(inputs)
        qkv = self.qkv_dw(qkv)

        if self.use_bias:
            k_bias = tf.zeros([self.channels], dtype=self.compute_dtype)
            use_bias = tf.concat([self.q_bias, k_bias, self.v_bias], axis=0)
            qkv = tf.nn.bias_add(qkv, use_bias)

        qkv = tf.reshape(qkv, [batch, height * width, self.num_heads, 3, self.channels // self.num_heads])
        qkv = tf.transpose(qkv, [0, 2, 1, 3, 4])
        q, k, v = tf.unstack(qkv, 3, axis=-2)

        q = tf.math.l2_normalize(q, axis=-1, epsilon=1.55e-5)
        k = tf.math.l2_normalize(k, axis=-1, epsilon=1.55e-5)

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
            'use_bias': self.use_bias
        })

        return config


@register_keras_serializable(package='SegMe>Policy>Backbone>DIY>CoMA')
class GGMSA(layers.Layer):
    def __init__(self, current_window, pretrain_window, num_heads, use_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        if current_window < pretrain_window:
            raise ValueError('Actual window size should not be less then pretrain one.')

        self.current_window = current_window
        self.pretrain_window = pretrain_window
        self.num_heads = num_heads
        self.use_bias = use_bias

    @shape_type_conversion
    def build(self, input_shape):
        self.channels = input_shape[-1]
        if self.channels is None:
            raise ValueError('Channel dimensions of the inputs should be defined. Found `None`.')
        if self.channels % self.num_heads:
            raise ValueError('Channel dimensions of the inputs should be a multiple of the number of heads.')

        self.window_length = self.current_window ** 2

        self.qkv = Conv(self.channels * 3, 1, use_bias=False, name='qkv')
        self.qkv_dw = ConvNorm(None, 3, use_bias=False, name='qkv_dw')  # From CvT

        if self.use_bias:
            self.q_bias = self.add_weight('q_bias', shape=[self.channels], initializer='zeros')
            self.v_bias = self.add_weight('v_bias', shape=[self.channels], initializer='zeros')

        self.scale = self.add_weight(
            'scale', shape=[self.num_heads, 1, 1],
            initializer=initializers.constant(np.log(10., dtype=self.dtype)),
            constraint=lambda s: tf.minimum(s, np.log(100., dtype=self.dtype)))

        self.cpb = Sequential([
            layers.Dense(512, name='cpb_fc0'),
            Act(name='cpb_act'),
            layers.Dense(self.num_heads, activation='sigmoid', use_bias=False, name='cpb_fc1')], name='cpb')

        self.proj = Conv(self.channels, 1, use_bias=False, name='proj')

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        qkv = self.qkv(inputs)
        qkv = self.qkv_dw(qkv)

        if self.use_bias:
            k_bias = tf.zeros([self.channels], dtype=self.compute_dtype)
            use_bias = tf.concat([self.q_bias, k_bias, self.v_bias], axis=0)
            qkv = tf.nn.bias_add(qkv, use_bias)

        outputs = with_partition_fused(self.qkv_attn, qkv, 'grid_size', self.current_window, 3, self.num_heads)

        outputs = self.proj(outputs)

        return outputs

    def qkv_attn(self, qkv, pad_size, pad_val):
        q, k, v = tf.unstack(qkv, 3, axis=-2)

        q = tf.math.l2_normalize(q, axis=-1, epsilon=1.55e-5)
        k = tf.math.l2_normalize(k, axis=-1, epsilon=1.55e-5)

        attn = tf.matmul(q * tf.exp(self.scale), k, transpose_b=True)
        attn = self.rel_bias(attn)
        attn = smart_cond(
            sum(pad_val) > 0,
            lambda: self.pad_mask(attn, pad_size, pad_val),
            lambda: tf.identity(attn))
        attn = tf.nn.softmax(attn)

        outputs = tf.matmul(attn, v)

        return outputs

    def rel_bias(self, attn):
        reltab = np.arange(1 - self.current_window, self.current_window).astype('float32')
        reltab = np.stack(np.meshgrid(reltab, reltab, indexing='ij'))
        reltab = np.transpose(reltab, [1, 2, 0])[None]
        reltab *= 8. / (self.pretrain_window - 1.)
        reltab = np.sign(reltab) * np.log1p(np.abs(reltab)) / np.log(8)
        reltab = tf.cast(reltab, self.compute_dtype)

        relidx = np.arange(self.current_window)
        relidx = np.stack(np.meshgrid(relidx, relidx, indexing='ij'), axis=0)
        relidx = np.reshape(relidx, [2, -1])
        relidx = relidx[:, :, None] - relidx[:, None]
        relidx = relidx + (self.current_window - 1)
        relidx = relidx[0] * (2 * self.current_window - 1) + relidx[1]
        relidx = np.reshape(relidx, [-1])
        relidx = tf.cast(relidx, 'int64')

        bias = self.cpb(reltab) * 16.
        bias = tf.reshape(bias, [-1, self.num_heads])
        bias = tf.gather(bias, relidx)
        bias = tf.reshape(bias, [self.window_length, self.window_length, -1])
        bias = tf.transpose(bias, perm=[2, 0, 1])[None]

        attn += bias

        return attn

    def pad_mask(self, attn, pad_size, pad_val):
        batch_size, pad_height, pad_width = pad_size
        src_height, src_width = pad_height - pad_val[0], pad_width - pad_val[1]

        hb_pad, wb_pad = pad_val[0] // 2, pad_val[1] // 2
        ha_pad, wa_pad = pad_val[0] - hb_pad, pad_val[1] - wb_pad
        paddings = [[0, 0], [hb_pad, ha_pad], [wb_pad, wa_pad], [0, 0]]

        mask = tf.ones((1, src_height, src_width, 1), dtype='int64')
        mask = tf.pad(mask, paddings)
        mask = partition_apply(mask, pad_height, pad_width, 'grid_size', self.current_window, 1)
        mask = tf.squeeze(mask == 0, axis=-1)[None, :, None, None]
        mask = -100. * tf.cast(mask, self.compute_dtype)

        num_windows = pad_height * pad_width // self.window_length
        attn = tf.reshape(attn, shape=[batch_size, num_windows, self.num_heads, self.window_length, self.window_length])
        attn += mask
        attn = tf.reshape(
            attn, shape=[batch_size * num_windows, self.num_heads, self.window_length, self.window_length])

        return attn

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()

        config.update({
            'current_window': self.current_window,
            'pretrain_window': self.pretrain_window,
            'num_heads': self.num_heads,
            'use_bias': self.use_bias
        })

        return config
