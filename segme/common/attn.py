import numpy as np
import tensorflow as tf
from keras import initializers, layers
from keras.saving.object_registration import register_keras_serializable
from keras.utils.control_flow_util import smart_cond
from keras.utils.tf_utils import shape_type_conversion
from segme.common.convnormact import ConvNorm, Conv, Act
from segme.common.pad import with_divisible_pad
from segme.common.part import partition_apply, partition_apply_fused, partition_reverse_fused
from segme.common.part import with_partition_fused, halo_partition, halo_partition_fused
from segme.common.sequent import Sequential


@register_keras_serializable(package='SegMe>Common')
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

        self.rel_bias = RelativeBias(
            self.current_window, self.pretrain_window, self.halo_window, self.num_heads, name='rel_bias')

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
            kv, pad_height, pad_width, self.current_window, self.halo_window, 2, self.num_heads, self.dilation_rate)

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
        attn += self.attn_mask(pad_size, pad_val)
        attn = tf.nn.softmax(attn)

        outputs = tf.matmul(attn, v)

        return outputs

    def attn_mask(self, pad_size, pad_val):
        return self.rel_bias(None) + self.pad_mask(pad_size, pad_val)

    def pad_mask(self, pad_size, pad_val):
        _, pad_height, pad_width = pad_size
        src_height = pad_height - sum(pad_val[:2])
        src_width = pad_width - sum(pad_val[2:])

        mask = tf.ones((1, src_height, src_width, 1), dtype=self.compute_dtype)
        mask = tf.pad(mask, [(0, 0), pad_val[:2], pad_val[2:], (0, 0)])
        mask = halo_partition(mask, pad_height, pad_width, self.current_window, self.halo_window, self.dilation_rate)
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
            'dilation_rate': self.dilation_rate,
            'use_bias': self.use_bias
        })

        return config


@register_keras_serializable(package='SegMe>Common')
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

        self.rel_bias = RelativeBias(
            self.current_window, self.pretrain_window, self.current_window, self.num_heads, name='rel_bias')

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
        attn += self.attn_mask(pad_size, pad_val)
        attn = tf.nn.softmax(attn)

        outputs = tf.matmul(attn, v)

        return outputs

    def attn_mask(self, pad_size, pad_val):
        mask = self.rel_bias(None)

        mask = smart_cond(
            sum(pad_val) > 0,
            lambda: mask + self.pad_mask(pad_size, pad_val),
            lambda: tf.identity(mask))

        return mask

    def pad_mask(self, pad_size, pad_val):
        _, pad_height, pad_width = pad_size
        src_height = pad_height - sum(pad_val[:2])
        src_width = pad_width - sum(pad_val[2:])

        mask = tf.ones((1, src_height, src_width, 1), dtype='int64')
        mask = tf.pad(mask, [(0, 0), pad_val[:2], pad_val[2:], (0, 0)])
        mask = partition_apply(mask, pad_height, pad_width, 'grid_size', self.current_window, 1)
        mask = tf.squeeze(mask == 0, axis=-1)[:, :, None, None]
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
            'use_bias': self.use_bias
        })

        return config


@register_keras_serializable(package='SegMe>Common')
class RelativeBias(layers.Layer):
    def __init__(self, query_window, pretrain_window, key_window, num_heads, **kwargs):
        super().__init__(**kwargs)
        if query_window < pretrain_window:
            raise ValueError('Actual window size should not be less then pretrain one.')

        if key_window < query_window:
            raise ValueError('Key window must be greater or equal to query one.')

        if (key_window - query_window) % 2:
            raise ValueError('Key window halo must be symmetric around query window.')

        self.query_window = query_window
        self.pretrain_window = pretrain_window
        self.key_window = key_window
        self.num_heads = num_heads

    def build(self, input_shape):
        key_halo = (self.key_window - self.query_window) // 2
        rel_tab = np.arange(1 - self.query_window - key_halo, self.query_window + key_halo).astype('float32')
        rel_tab = np.stack(np.meshgrid(rel_tab, rel_tab, indexing='ij'))
        rel_tab = np.transpose(rel_tab, [1, 2, 0])[None]
        rel_tab *= 8. / (self.pretrain_window - 1.)
        rel_tab = np.sign(rel_tab) * np.log1p(np.abs(rel_tab)) / np.log(8)
        rel_tab = rel_tab.reshape([-1, 2])
        self.rel_tab = tf.cast(rel_tab, self.compute_dtype)

        query_idx = np.arange(self.query_window)
        query_idx = np.stack(np.meshgrid(query_idx, query_idx, indexing='ij'), axis=0)
        query_idx = np.reshape(query_idx, [2, -1])
        key_idx = np.arange(self.key_window)
        key_idx = np.stack(np.meshgrid(key_idx, key_idx, indexing='ij'), axis=0)
        key_idx = np.reshape(key_idx, [2, -1])
        rel_idx = query_idx[:, :, None] - key_idx[:, None]
        rel_idx = rel_idx + (self.key_window - 1)
        rel_idx = rel_idx[0] * (self.query_window + self.key_window - 1) + rel_idx[1]
        rel_idx = np.reshape(rel_idx, [-1])
        self.rel_idx = tf.cast(rel_idx, 'int64')

        self.cpb = Sequential([
            layers.Dense(512, name='cpb_expand'),
            Act(name='cpb_act'),
            layers.Dense(self.num_heads, activation='sigmoid', use_bias=False, name='cpb_squeeze')], name='cpb')

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = self.cpb(self.rel_tab) * 16.
        outputs = tf.gather(outputs, self.rel_idx)
        outputs = tf.reshape(outputs, [self.query_window ** 2, self.key_window ** 2, self.num_heads])
        outputs = tf.transpose(outputs, perm=[2, 0, 1])[None, None]

        return outputs

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([1, 1, self.num_heads, self.query_window ** 2, self.key_window ** 2])

    def get_config(self):
        config = super().get_config()

        config.update({
            'query_window': self.query_window,
            'pretrain_window': self.pretrain_window,
            'key_window': self.key_window,
            'num_heads': self.num_heads
        })

        return config


@register_keras_serializable(package='SegMe>Common')
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

        if 1 == self.num_heads:
            qkv = tf.reshape(qkv, [batch, 1, height * width, 3, self.channels])
        else:
            qkv = tf.reshape(qkv, [batch, height * width, self.num_heads, 3, self.channels // self.num_heads])
            qkv = tf.transpose(qkv, [0, 2, 1, 3, 4])
        q, k, v = tf.unstack(qkv, 3, axis=-2)

        q = tf.math.l2_normalize(q, axis=-2, epsilon=1.55e-5)  # TODO
        k = tf.math.l2_normalize(k, axis=-2, epsilon=1.55e-5)

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
