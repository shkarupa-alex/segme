import tensorflow as tf
from keras import layers
from keras.saving import register_keras_serializable
from keras.src.utils.tf_utils import shape_type_conversion
from segme.common.convnormact import Norm


@register_keras_serializable(package='SegMe>Common')
class MultiHeadAttentionPooling(layers.Layer):
    def __init__(self, heads, queries, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(min_ndim=3)
        self.heads = heads
        self.queries = queries

    @shape_type_conversion
    def build(self, input_shape):
        # noinspection PyAttributeOutsideInit
        self.channels = input_shape[-1]
        if self.channels is None:
            raise ValueError('Channel dimensions of the inputs should be defined. Found `None`.')
        self.input_spec = layers.InputSpec(min_ndim=3, axes={-1: self.channels})

        # noinspection PyAttributeOutsideInit
        self.query = self.add_weight(name='query', shape=(1, self.queries, self.channels))

        # noinspection PyAttributeOutsideInit
        self.mhsa = layers.MultiHeadAttention(self.heads, self.channels // self.heads, name='mhsa')

        # noinspection PyAttributeOutsideInit
        self.ln_q = Norm(policy='conv-ln1em5-relu', name='ln_q')

        # noinspection PyAttributeOutsideInit
        self.ln_k = Norm(policy='conv-ln1em5-relu', name='ln_k')

        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        batch = tf.shape(inputs)[0]

        q = self.ln_q(self.query)
        q = tf.repeat(q, batch, axis=0)

        k = self.ln_k(inputs)
        k = tf.reshape(k, [batch, -1, self.channels])
        x = self.mhsa(q, k)

        return x

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[:1] + (self.queries,) + input_shape[-1:]

    def get_config(self):
        config = super().get_config()
        config.update({
            'heads': self.heads,
            'queries': self.queries
        })

        return config
