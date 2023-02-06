import tensorflow as tf
from keras import backend, layers
from keras.utils.control_flow_util import smart_cond
from keras.saving.object_registration import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion


@register_keras_serializable(package='SegMe>Common')
class DropPath(layers.Layer):
    def __init__(self, rate, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(min_ndim=1)

        if not 0. <= rate <= 1.:
            raise ValueError(f'Invalid value {rate} received for `rate`. Expected a value between 0 and 1.')

        self.rate = rate

    def call(self, inputs, training=None, **kwargs):
        if 0. == self.rate:
            return inputs

        if training is None:
            training = backend.learning_phase()

        outputs = smart_cond(training, lambda: self.drop(inputs), lambda: tf.identity(inputs))

        return outputs

    def drop(self, inputs):
        keep = 1.0 - self.rate
        batch = tf.shape(inputs)[0]
        shape = [batch] + [1] * (inputs.shape.rank - 1)

        random = tf.random.uniform(shape, dtype=self.compute_dtype)
        random = tf.floor(random + keep, self.compute_dtype) / keep

        outputs = inputs * random

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({'rate': self.rate})

        return config


@register_keras_serializable(package='SegMe>Common')
class DropBlock(layers.Layer):
    """ Proposed in: https://arxiv.org/pdf/1810.12890.pdf """

    def __init__(self, rate, size, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        if not 0. <= rate <= 1.:
            raise ValueError(f'Invalid value {rate} received for `rate`. Expected a value between 0 and 1.')
        if size < 1:
            raise ValueError(f'Invalid value {size} received for `size`. Expected a value above 0.')

        self.rate = rate
        self.size = size

    def call(self, inputs, training=None, **kwargs):
        if 0. == self.rate:
            return inputs

        if training is None:
            training = backend.learning_phase()

        outputs = smart_cond(training, lambda: self.drop(inputs), lambda: tf.identity(inputs))

        return outputs

    def drop(self, inputs):
        gamma = self.rate / self.size ** 2

        mask = tf.random.uniform(tf.shape(inputs), dtype=self.compute_dtype)
        mask = tf.cast(mask < gamma, self.compute_dtype)
        mask = 1. - tf.nn.max_pool2d(mask, self.size, 1, 'SAME')
        mask = mask / tf.reduce_mean(mask, axis=[1, 2], keepdims=True)

        outputs = inputs * mask

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            'rate': self.rate,
            'size': self.size
        })

        return config
