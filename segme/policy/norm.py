import math
import tensorflow as tf
from keras import constraints, initializers, layers, regularizers
from keras.utils.conv_utils import normalize_data_format
from keras.saving.object_registration import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion, validate_axis
from segme.policy.registry import LayerRegistry

NORMALIZATIONS = LayerRegistry()
NORMALIZATIONS.register('brn')({'class_name': 'SegMe>Policy>Normalization>BatchNorm', 'config': {
    'renorm': True, 'renorm_clipping': {'rmax': 3, 'rmin': 0.3333, 'dmax': 5}, 'renorm_momentum': 0.99}})


@NORMALIZATIONS.register('bn')
@register_keras_serializable(package='SegMe>Policy>Normalization')
class BatchNorm(layers.BatchNormalization):
    """Overload for data_format understanding and fused-epsilon check"""

    def __init__(self, data_format=None, momentum=0.99, epsilon=1e-3, center=True, scale=True, beta_initializer='zeros',
                 gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones',
                 beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None,
                 renorm=False, renorm_clipping=None, renorm_momentum=0.99, fused=None, trainable=True,
                 virtual_batch_size=None, adjustment=None, name=None, **kwargs):
        self.data_format = normalize_data_format(data_format)
        axis = -1 if 'channels_last' == self.data_format else 1
        super().__init__(
            axis=axis, momentum=momentum, epsilon=epsilon, center=center, scale=scale,
            beta_initializer=beta_initializer, gamma_initializer=gamma_initializer,
            moving_mean_initializer=moving_mean_initializer, moving_variance_initializer=moving_variance_initializer,
            beta_regularizer=beta_regularizer, gamma_regularizer=gamma_regularizer, beta_constraint=beta_constraint,
            gamma_constraint=gamma_constraint, renorm=renorm, renorm_clipping=renorm_clipping,
            renorm_momentum=renorm_momentum, fused=fused, trainable=trainable, virtual_batch_size=virtual_batch_size,
            adjustment=adjustment, name=name, **kwargs)

    def _raise_if_fused_cannot_be_used(self):
        if self.epsilon < 1.001e-5:
            raise ValueError(
                f'Fused batch normalization is not supported for epsilon {self.epsilon} (<1.001e-5).')

        super()._raise_if_fused_cannot_be_used()

    def get_config(self):
        config = super().get_config()
        config.update({'data_format': self.data_format})

        del config['axis']

        return config


@NORMALIZATIONS.register('ln')
@register_keras_serializable(package='SegMe>Policy>Normalization')
class LayerNorm(layers.LayerNormalization):
    """Overload for data_format understanding and fused implementation"""

    def __init__(self, data_format=None, epsilon=1e-3, center=True, scale=True, beta_initializer='zeros',
                 gamma_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
                 gamma_constraint=None, fused=None, **kwargs):
        self.data_format = normalize_data_format(data_format)
        axis = -1 if 'channels_last' == self.data_format else 1
        super().__init__(
            axis=axis, epsilon=epsilon, center=center, scale=scale, beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer, beta_regularizer=beta_regularizer, gamma_regularizer=gamma_regularizer,
            beta_constraint=beta_constraint, gamma_constraint=gamma_constraint, **kwargs)
        self.fused = fused

    def _fused_can_be_used(self, ndims):
        if self.fused is False:
            return False

        if 'float32' != self.dtype:
            if self.fused is None:
                return False
            raise ValueError(
                f'Fused layer normalization is only supported when the variables dtype is '
                f'float32. Got dtype: {self.dtype}.')

        if self._compute_dtype not in ('float16', 'float16', 'float32', None):
            if self.fused is None:
                return False
            raise ValueError(
                f'Fused layer normalization is only supported when the compute dtype is '
                f'float16, bfloat16, or float32. Got dtype: {self._compute_dtype}.')

        if self.epsilon < 1.001e-5:
            if self.fused is None:
                return False
            raise ValueError(
                f'Fused layer normalization is not supported for epsilon {self.epsilon} (<1.001e-5).')

        if self.axis[0] != ndims - 1:
            if self.fused is None:
                return False
            raise ValueError(
                f'Fused layer normalization is not supported for axis {self.axis} and inputs rank {ndims}.')

        return True

    def get_config(self):
        config = super().get_config()
        config.update({
            'data_format': self.data_format,
            'fused': self.fused
        })

        del config['axis']

        return config


@NORMALIZATIONS.register('gn')
@register_keras_serializable(package='SegMe>Policy>Normalization')
class GroupNorm(layers.GroupNormalization):
    """Overload for fused implementation, groups estimation and BCHW issue fix"""

    def __init__(self, groups=None, data_format=None, epsilon=1e-3, center=True, scale=True, beta_initializer='zeros',
                 gamma_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
                 gamma_constraint=None, **kwargs):
        self.data_format = normalize_data_format(data_format)
        axis = -1 if 'channels_last' == self.data_format else 1
        kwargs['autocast'] = False
        kwargs['dtype'] = 'float32'
        super().__init__(
            groups=-1, axis=axis, epsilon=epsilon, center=center, scale=scale, beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer, beta_regularizer=beta_regularizer, gamma_regularizer=gamma_regularizer,
            beta_constraint=beta_constraint, gamma_constraint=gamma_constraint, **kwargs)

        self._groups = groups

    @shape_type_conversion
    def build(self, input_shape):
        self.rank = len(input_shape)
        self.channels = input_shape[self.axis]
        if self.channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')

        if self._groups is None:
            # Best results in paper obtained with groups=32 or channels_per_group=16
            best_cpg = self.channels // (2 ** math.floor(math.log2(self.channels) / 2))
            best_grp = min(32, self.channels // min(16, best_cpg))
            if self.channels % best_grp:
                raise ValueError(
                    f'Unable to choose best number of groups for the number of channels ({self.channels}). '
                    f'It supposed to be somewhere near {best_grp}.')

            self.groups = best_grp
        elif -1 == self._groups:
            self.groups = self.channels
        else:
            self.groups = self._groups
            if self.channels < self.groups:
                raise ValueError(
                    f'Number of groups ({self.groups}) cannot be more than the number of channels ({self.channels}).')

            if self.channels % self.groups:
                raise ValueError(
                    f'Number of groups ({self.groups}) must be a multiple of the number of channels ({self.channels}).')

        super().build(input_shape)

    def call(self, inputs):
        outputs = tf.cast(inputs, self.dtype)
        outputs = super().call(outputs)
        outputs = tf.saturate_cast(outputs, inputs.dtype)

        return outputs

    def _reshape_into_groups(self, inputs):
        input_shape = tf.shape(inputs)
        group_shape = tf.unstack(input_shape)
        group_shape[self.axis] = input_shape[self.axis] // self.groups
        group_shape.insert(self.axis, self.groups)

        return tf.reshape(inputs, group_shape)

    def _create_broadcast_shape(self, input_shape):
        del input_shape

        broadcast_shape = [1] * self.rank
        broadcast_shape[self.axis] = self.channels // self.groups
        broadcast_shape.insert(self.axis, self.groups)

        return broadcast_shape

    def compute_output_signature(self, input_signature):
        return input_signature

    def get_config(self):
        config = super().get_config()
        config.update({
            'data_format': self.data_format,
            'groups': self._groups
        })

        del config['axis']

        return config


@NORMALIZATIONS.register('frn')
@register_keras_serializable(package='SegMe>Policy>Normalization')
class FilterResponseNorm(layers.Layer):
    """Rewrite for any shape support"""

    def __init__(self, data_format=None, epsilon=1e-6, beta_initializer='zeros', gamma_initializer='ones',
                 beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None,
                 learned_epsilon=False, **kwargs):
        kwargs['autocast'] = False
        kwargs['dtype'] = 'float32'
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(min_ndim=3)
        self.supports_masking = True

        self.data_format = normalize_data_format(data_format)
        self.epsilon = epsilon
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)
        self.use_eps_learned = learned_epsilon

    @shape_type_conversion
    def build(self, input_shape):
        channels = input_shape[-1]
        if channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')

        axis = list(range(1, len(input_shape)))
        if 'channels_last' == self.data_format:
            self.axis = axis[:-1]
        else:
            self.axis = axis[1:]

        self.gamma = self.add_weight(
            shape=[1, 1, 1, channels], name='gamma', initializer=self.gamma_initializer,
            regularizer=self.gamma_regularizer, constraint=self.gamma_constraint)
        self.beta = self.add_weight(
            shape=[1, 1, 1, channels], name='beta', initializer=self.beta_initializer,
            regularizer=self.beta_regularizer, constraint=self.beta_constraint)

        if self.use_eps_learned:
            self.eps_learned = self.add_weight(
                shape=(1,), name='learned_epsilon', initializer=initializers.Constant(1e-4),
                constraint=lambda e: tf.minimum(e, 0.))

        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        outputs = tf.cast(inputs, self.dtype)

        epsilon = self.epsilon
        if self.use_eps_learned:
            epsilon += tf.math.abs(self.eps_learned)

        nu2 = tf.reduce_mean(tf.square(outputs), axis=self.axis, keepdims=True)
        outputs = outputs * tf.math.rsqrt(nu2 + epsilon) * self.gamma + self.beta

        outputs = tf.saturate_cast(outputs, inputs.dtype)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_output_signature(self, input_signature):
        return input_signature

    def get_config(self):
        config = super().get_config()
        config.update({
            'data_format': self.data_format,
            'epsilon': self.epsilon,
            'learned_epsilon': self.use_eps_learned,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        })

        return config
