import tensorflow as tf
import warnings
from keras import layers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from tensorflow_addons import layers as add_layers
from segme.registry import LayerRegistry

NORMALIZATIONS = LayerRegistry()
NORMALIZATIONS.register('bn')(layers.BatchNormalization)

@NORMALIZATIONS.register('ln')
@register_keras_serializable(package='SegMe>Policy>Normalization')
class LayerNormalization(layers.LayerNormalization):
    """Overload casting for stability and fused implementation"""

    def __init__(self, axis=-1, epsilon=1e-3, center=True, scale=True, beta_initializer='zeros',
                 gamma_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
                 gamma_constraint=None, dtype='float32', **kwargs):
        kwargs['autocast'] = False
        super().__init__(axis=axis, epsilon=epsilon, center=center, scale=scale, beta_initializer=beta_initializer,
                         gamma_initializer=gamma_initializer, beta_regularizer=beta_regularizer,
                         gamma_regularizer=gamma_regularizer, beta_constraint=beta_constraint,
                         gamma_constraint=gamma_constraint, dtype=dtype, **kwargs)

    @shape_type_conversion
    def build(self, input_shape):
        super().build(input_shape)
        if not self._fused:
            warnings.warn(f'Layer {self.name} will use an inefficient implementation.')

    def call(self, inputs, *args, **kwargs):
        cast_input = inputs.dtype != tf.dtypes.float32

        outputs = inputs
        if cast_input:
            outputs = tf.cast(inputs, 'float32')

        outputs = super().call(outputs)

        if cast_input:
            outputs = tf.saturate_cast(outputs, inputs.dtype)

        return outputs

    def compute_output_signature(self, input_signature):
        return input_signature


@NORMALIZATIONS.register('gn')
@register_keras_serializable(package='SegMe>Policy>Normalization')
class GroupNormalization(add_layers.GroupNormalization):
    """Overload casting for stability and groups reduction"""

    def __init__(self, groups=None, axis=-1, epsilon=1e-3, center=True, scale=True, beta_initializer='zeros',
                 gamma_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
                 gamma_constraint=None, dtype='float32', **kwargs):
        kwargs['autocast'] = False
        super().__init__(groups=32, axis=axis, epsilon=epsilon, center=center, scale=scale,
                         beta_initializer=beta_initializer, gamma_initializer=gamma_initializer,
                         beta_regularizer=beta_regularizer, gamma_regularizer=gamma_regularizer,
                         beta_constraint=beta_constraint, gamma_constraint=gamma_constraint, dtype=dtype, **kwargs)

        self._groups = groups

    @shape_type_conversion
    def build(self, input_shape):
        channels = input_shape[-1]
        if channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')

        if self._groups is None:
            self.groups = min(self.groups, channels)

        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        cast_input = inputs.dtype != tf.dtypes.float32

        outputs = inputs
        if cast_input:
            outputs = tf.cast(inputs, 'float32')

        outputs = super().call(outputs)

        if cast_input:
            outputs = tf.saturate_cast(outputs, inputs.dtype)

        return outputs

    def compute_output_signature(self, input_signature):
        return input_signature

    def get_config(self):
        config = super().get_config()
        config.update({'groups': self._groups})

        return config


@NORMALIZATIONS.register('frn')
@register_keras_serializable(package='SegMe>Policy>Normalization')
class FilterResponseNormalization(add_layers.FilterResponseNormalization):
    """Overload casting for stability"""

    def __init__(self, epsilon=1e-6, axis=(1, 2), beta_initializer='zeros', gamma_initializer='ones',
                 beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None,
                 learned_epsilon=False, learned_epsilon_constraint=None, dtype='float32', **kwargs):
        kwargs['autocast'] = False
        super().__init__(epsilon=epsilon, axis=axis, beta_initializer=beta_initializer,
                         gamma_initializer=gamma_initializer, beta_regularizer=beta_regularizer,
                         gamma_regularizer=gamma_regularizer, beta_constraint=beta_constraint,
                         gamma_constraint=gamma_constraint, learned_epsilon=learned_epsilon,
                         learned_epsilon_constraint=learned_epsilon_constraint, dtype=dtype, **kwargs)

    def call(self, inputs, *args, **kwargs):
        cast_input = inputs.dtype != tf.dtypes.float32

        outputs = inputs
        if cast_input:
            outputs = tf.cast(inputs, 'float32')

        outputs = super().call(outputs)

        if cast_input:
            outputs = tf.saturate_cast(outputs, inputs.dtype)

        return outputs

    def _check_axis(self, axis):
        if isinstance(axis, tuple):
            axis = list(axis)

        super()._check_axis(axis)

    def compute_output_signature(self, input_signature):
        return input_signature
