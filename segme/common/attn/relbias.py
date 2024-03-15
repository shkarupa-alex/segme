import numpy as np
import tensorflow as tf
from tf_keras import layers
from tf_keras.saving import register_keras_serializable
from segme.common.sequence import Sequence


@register_keras_serializable(package='SegMe>Common')
class RelativeBias(layers.Layer):
    def __init__(self, query_window, key_window, pretrain_window, num_heads, cpb_units=512, **kwargs):
        super().__init__(**kwargs)
        if key_window < query_window:
            raise ValueError('Key window must be greater or equal to query one.')

        if (key_window - query_window) % 2:
            raise ValueError('Key window halo must be symmetric around query window.')

        self.query_window = query_window
        self.key_window = key_window
        self.pretrain_window = pretrain_window
        self.num_heads = num_heads
        self.cpb_units = cpb_units

    def build(self, input_shape):
        key_halo = (self.key_window - self.query_window) // 2
        rel_tab = np.arange(1 - self.query_window - key_halo, self.query_window + key_halo).astype('float32')
        rel_tab = np.stack(np.meshgrid(rel_tab, rel_tab, indexing='ij'))
        rel_tab = np.transpose(rel_tab, [1, 2, 0])[None]
        rel_tab *= 8. / (self.pretrain_window - 1.)
        rel_tab = np.sign(rel_tab) * np.log1p(np.abs(rel_tab)) / np.log(8)
        rel_tab = np.reshape(rel_tab, [-1, 2])
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
        self.rel_idx = tf.cast(rel_idx, 'int32')

        self.cpb = Sequence([
            layers.Dense(self.cpb_units, activation='relu', kernel_initializer='he_uniform', name='expand'),
            layers.Dense(self.num_heads, activation='sigmoid', use_bias=False, name='squeeze')
        ], name='cpb')

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
            'key_window': self.key_window,
            'pretrain_window': self.pretrain_window,
            'num_heads': self.num_heads,
            'cpb_units': self.cpb_units
        })

        return config
