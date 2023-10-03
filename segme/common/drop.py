import tensorflow as tf
from keras import backend, layers
from keras.saving import register_keras_serializable
from keras.src.utils.control_flow_util import smart_cond
from keras.src.utils.tf_utils import shape_type_conversion
from segme.common.shape import get_shape


@register_keras_serializable(package='SegMe>Common')
class DropPath(layers.Dropout):
    def __init__(self, rate, seed=None, **kwargs):
        kwargs.pop('noise_shape', None)
        super().__init__(rate=rate, seed=seed, **kwargs)

    def _get_noise_shape(self, inputs):
        batch_size, _ = get_shape(inputs, axis=[0])
        noise_shape = batch_size + [1] * (inputs.shape.rank - 1)
        noise_shape = tf.convert_to_tensor(noise_shape)

        return noise_shape

    def get_config(self):
        config = super().get_config()
        del config['noise_shape']

        return config


@register_keras_serializable(package='SegMe>Common')
class SlicePath(layers.Layer):
    def __init__(self, rate, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(min_ndim=1)

        if not 0. <= rate <= 1.:
            raise ValueError(f'Invalid value {rate} received for `rate`. Expected a value between 0 and 1.')

        self.rate = rate
        self.seed = seed

    def call(self, inputs, training=None, **kwargs):
        [batch_size], _ = get_shape(inputs, axis=[0])

        if 0. == self.rate:
            return inputs, tf.ones([batch_size], dtype='bool')

        if training is None:
            training = backend.learning_phase()

        outputs, slice_mask = smart_cond(
            training,
            lambda: self.maybe_slice(inputs, batch_size),
            lambda: (tf.identity(inputs), tf.ones([batch_size], dtype='bool')))

        return outputs, slice_mask

    def maybe_slice(self, inputs, batch_size):
        batch_size = tf.convert_to_tensor(batch_size, 'int32')
        keep_size = tf.cast(batch_size, 'float32') * (1. - self.rate)
        keep_size = tf.math.ceil(keep_size / 8.) * 8.
        keep_size = tf.cast(keep_size, batch_size.dtype)
        keep_size = tf.minimum(keep_size, batch_size)

        outputs, slice_mask = smart_cond(
            keep_size < batch_size,
            lambda: self.apply_slice(inputs, batch_size, keep_size),
            lambda: (tf.identity(inputs), tf.ones([batch_size], dtype='bool')))

        return outputs, slice_mask

    def apply_slice(self, inputs, batch_size, keep_size):
        keep_mask = tf.concat([
            tf.ones([keep_size], dtype='bool'),
            tf.zeros([batch_size - keep_size], dtype='bool')
        ], axis=-1)
        keep_mask = tf.random.shuffle(keep_mask, self.seed)

        outputs = inputs[keep_mask]

        return outputs, keep_mask

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return (None,) + input_shape[1:], input_shape[:1]

    def compute_output_signature(self, input_signature):
        output_signature, mask_signature = super().compute_output_signature(input_signature)
        mask_signature = tf.TensorSpec(dtype='bool', shape=mask_signature.shape)

        return output_signature, mask_signature

    def get_config(self):
        config = super().get_config()
        config.update({
            'rate': self.rate,
            'seed': self.seed
        })

        return config


@register_keras_serializable(package='SegMe>Common')
class RestorePath(layers.Layer):
    def __init__(self, rate, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [layers.InputSpec(min_ndim=1), layers.InputSpec(ndim=1, dtype='bool')]

        self.rate = rate
        self.seed = seed

    def call(self, inputs, training=None, **kwargs):
        outputs, keep_mask = inputs

        if 0. == self.rate:
            return outputs

        if training is None:
            training = backend.learning_phase()

        outputs = smart_cond(training, lambda: self.maybe_restore(outputs, keep_mask), lambda: tf.identity(outputs))

        return outputs

    def maybe_restore(self, inputs, keep_mask):
        inputs_shape, _ = get_shape(inputs)
        keep_size = inputs_shape[0]
        batch_size = tf.size(keep_mask)

        keep_up = tf.cast(batch_size, 'float32') / tf.cast(keep_size, 'float32')
        keep_min = (1. - self.rate) * keep_up
        keep_max = (2. - self.rate) * keep_up
        noise_shape = [keep_size] + [1] * (inputs.shape.rank - 1)
        random_mask = tf.random.uniform(
            noise_shape, minval=keep_min, maxval=keep_max, dtype='float32', seed=self.seed)

        inv_keep = 1. / (1. - self.rate)
        random_mask = tf.cast(random_mask >= 1., inputs.dtype) * inv_keep

        outputs = inputs * random_mask

        outputs = smart_cond(
            tf.equal(batch_size, keep_size),
            lambda: tf.identity(outputs),
            lambda: self.apply_restore(outputs, batch_size, inputs_shape, keep_mask))

        return outputs

    def apply_restore(self, inputs, batch_size, inputs_shape, keep_mask):
        zeros = tf.zeros([1] + inputs_shape[1:], dtype=inputs.dtype)
        outputs = tf.concat([zeros, inputs], axis=0)

        indices = tf.range(1, batch_size + 1, dtype='int32')
        indices -= tf.cumsum(tf.cast(~keep_mask, 'int32'))
        indices = tf.where(keep_mask, indices, 0)
        outputs = tf.gather(outputs, indices, axis=0)

        # TODO: Reading input as constant from a dynamic tensor is not yet supported
        # from tensorflow.python.ops import data_flow_ops
        #
        # join_indices = tf.range(batch_size, dtype='int32')
        #
        # zero_shape = [batch_size - inputs_shape[0]] + inputs_shape[1:]
        # zero_inputs = tf.zeros(zero_shape, dtype=inputs.dtype)
        #
        # outputs = data_flow_ops.parallel_dynamic_stitch(
        #     [join_indices[keep_mask], join_indices[~keep_mask]],
        #     [inputs, zero_inputs]
        # )

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[1] + input_shape[0][1:]

    def get_config(self):
        config = super().get_config()
        config.update({
            'rate': self.rate,
            'seed': self.seed
        })

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

        shape, _ = get_shape(inputs)
        mask = tf.random.uniform(shape, dtype=self.compute_dtype)
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
