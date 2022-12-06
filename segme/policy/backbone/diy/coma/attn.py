import numpy as np
import tensorflow as tf
from keras import initializers, layers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from segme.common.convnormact import ConvNorm, Conv, Act
from segme.common.pad import with_divisible_pad
from segme.common.sequent import Sequential
from segme.policy.backbone.diy.coma.part import partition_apply, partition_reverse


@register_keras_serializable(package='SegMe>Policy>Backbone>DIY>CoMA')
class DHMSA(layers.Layer):
    def __init__(self, current_window, pretrain_window, num_heads, dilation_rate=1, qkv_bias=True, attn_drop=0.,
                 proj_drop=0., **kwargs):
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
        self.qkv_bias = qkv_bias
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop

    @shape_type_conversion
    def build(self, input_shape):
        self.channels = input_shape[-1]
        if self.channels is None:
            raise ValueError('Channel dimensions of the inputs should be defined. Found `None`.')

        self.window_length = self.current_window ** 2
        self.halo_window = self.current_window * 2
        self.halo_length = self.halo_window ** 2
        self.halo_kernel = [1, self.halo_window, self.halo_window, 1]
        self.halo_stride = [1, self.current_window, self.current_window, 1]
        self.halo_dirate = [1, self.dilation_rate, self.dilation_rate, 1]

        self.qkv = Sequential([
            ConvNorm(None, 3, name='qkv_dw'),  # From CvT
            Conv(self.channels * 3, 1, use_bias=False, name='qkv_pw')], name='qkv')

        if self.qkv_bias:
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

        self.drop_attn = layers.Dropout(self.attn_drop, name='attn_drop')
        self.proj = Conv(self.channels, 1, name='proj')
        self.drop_proj = layers.Dropout(self.proj_drop, name='proj_drop')

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        qkv = self.qkv(inputs)
        if self.qkv_bias:
            k_bias = tf.zeros([self.channels], dtype=self.compute_dtype)
            qkv_bias = tf.concat([self.q_bias, k_bias, self.v_bias], axis=0)
            qkv = tf.nn.bias_add(qkv, qkv_bias)

        outputs = with_divisible_pad(self.qkv_part, qkv, self.current_window * self.dilation_rate)

        outputs = self.proj(outputs)
        outputs = self.drop_proj(outputs)

        return outputs

    def qkv_part(self, qkv, with_pad, pad_size, pad_val):
        _, height, width = pad_size
        halo_size = height * 2, width * 2

        q, kv = tf.split(qkv, [self.channels, self.channels * 2], axis=-1)
        q_part = partition_apply(q, height, width, 'window_size', self.current_window, self.dilation_rate)
        kv_part = self.halo_part(kv, halo_size)

        parted = self.qkv_attn(
            q_part, kv_part, with_pad=with_pad, pad_size=pad_size, pad_val=pad_val, halo_size=halo_size)
        parted = partition_reverse(parted, height, width, 'window_size', self.current_window, self.dilation_rate)

        return parted

    def halo_part(self, x, halo_size):
        # From HaloNet
        halo_height, halo_width = halo_size
        halo_blocks = [halo_height // self.halo_window, halo_width // self.halo_window]
        channel_size = x.shape[-1]

        x = tf.image.extract_patches(x, self.halo_kernel, self.halo_stride, self.halo_dirate, padding='SAME')
        x = tf.reshape(x, [-1, *halo_blocks, self.halo_window, self.halo_window, channel_size])
        x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
        x = tf.reshape(x, [-1, halo_height, halo_width, channel_size])
        x = partition_apply(x, halo_height, halo_width, 'window_size', self.halo_window, self.dilation_rate)

        return x

    def qkv_attn(self, q, kv, with_pad, pad_size, pad_val, halo_size):
        del with_pad

        q = tf.reshape(q, [-1, self.window_length, self.num_heads, self.channels // self.num_heads])
        q = tf.transpose(q, [0, 2, 1, 3])

        kv = tf.reshape(kv, [-1, self.halo_length, 2, self.num_heads, self.channels // self.num_heads])
        kv = tf.transpose(kv, [2, 0, 3, 1, 4])
        k, v = tf.unstack(kv, 2, axis=0)

        q = tf.math.l2_normalize(q, axis=-1)
        k = tf.math.l2_normalize(k, axis=-1)

        attn = tf.matmul(q * tf.exp(self.scale), k, transpose_b=True)
        attn = self.rel_bias(attn)
        attn = self.pad_mask(attn, pad_size, pad_val, halo_size)
        attn = tf.nn.softmax(attn)
        attn = self.drop_attn(attn)

        outputs = tf.transpose(tf.matmul(attn, v), perm=[0, 2, 1, 3])
        outputs = tf.reshape(outputs, [-1, self.window_length, self.channels])

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

    def pad_mask(self, attn, pad_size, pad_val, halo_size):
        batch_size, pad_height, pad_width = pad_size
        src_height, src_width = pad_height - pad_val[0], pad_width - pad_val[1]
        halo_height, halo_width = halo_size

        mask = tf.ones((1, src_height, src_width, 1), dtype='int32')
        mask = tf.pad(mask, [(0, 0), (0, pad_val[0]), (0, pad_val[1]), (0, 0)])
        mask = self.halo_part(mask, halo_size)
        mask = tf.squeeze(mask == 0, axis=-1)[None, :, None, None]
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
            'qkv_bias': self.qkv_bias,
            'attn_drop': self.attn_drop,
            'proj_drop': self.proj_drop,
        })

        return config


@register_keras_serializable(package='SegMe>Policy>Backbone>DIY>CoMA')
class CHMSA(layers.Layer):
    def __init__(self, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0., **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop

    @shape_type_conversion
    def build(self, input_shape):
        self.channels = input_shape[-1]
        if self.channels is None:
            raise ValueError('Channel dimensions of the inputs should be defined. Found `None`.')

        self.qkv = Sequential([
            ConvNorm(None, 3, name='qkv_dw'),  # From CvT
            Conv(self.channels * 3, 1, use_bias=False, name='qkv_pw')], name='qkv')
        if self.qkv_bias:
            self.q_bias = self.add_weight('q_bias', shape=[self.channels], initializer='zeros')
            self.v_bias = self.add_weight('v_bias', shape=[self.channels], initializer='zeros')

        self.scale = self.add_weight(
            'scale', shape=[self.num_heads, 1, 1],
            initializer=initializers.constant(np.log(10., dtype=self.dtype)),
            constraint=lambda s: tf.minimum(s, np.log(100., dtype=self.dtype)))

        self.drop_attn = layers.Dropout(self.attn_drop, name='attn_drop')
        self.proj = layers.Dense(self.channels, name='proj')
        self.drop_proj = layers.Dropout(self.proj_drop, name='proj_drop')

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        batch, height, width, _ = tf.unstack(tf.shape(inputs))

        qkv = self.qkv(inputs)
        if self.qkv_bias:
            k_bias = tf.zeros([self.channels], dtype=self.compute_dtype)
            qkv_bias = tf.concat([self.q_bias, k_bias, self.v_bias], axis=0)
            qkv = tf.nn.bias_add(qkv, qkv_bias)

        qkv = tf.reshape(qkv, [batch, height * width, 3, self.num_heads, self.channels // self.num_heads])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])
        q, k, v = tf.unstack(qkv, 3)

        q = tf.math.l2_normalize(q, axis=-1)
        k = tf.math.l2_normalize(k, axis=-1)

        attn = tf.matmul(q * tf.exp(self.scale), k, transpose_a=True)
        attn = tf.nn.softmax(attn)
        attn = self.drop_attn(attn)

        outputs = tf.transpose(tf.matmul(attn, v, transpose_b=True), perm=[0, 3, 1, 2])
        outputs = tf.reshape(outputs, [batch, height, width, self.channels])

        outputs = self.proj(outputs)
        outputs = self.drop_proj(outputs)

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
            'attn_drop': self.attn_drop,
            'proj_drop': self.proj_drop,
        })

        return config
