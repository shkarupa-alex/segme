import math
import tensorflow as tf
import warnings
from keras import layers
from keras.utils.conv_utils import normalize_data_format
from keras.saving.object_registration import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from tensorflow_addons import layers as add_layers
from segme.policy.registry import LayerRegistry

NORMALIZATIONS = LayerRegistry()
NORMALIZATIONS.register('brn')({'class_name': 'SegMe>Policy>Normalization>BatchNorm', 'config': {
    'renorm': True, 'renorm_clipping': {'rmax': 3, 'rmin': 0.3333, 'dmax': 5}, 'renorm_momentum': 0.99}})


@NORMALIZATIONS.register('bn')
@register_keras_serializable(package='SegMe>Policy>Normalization')
class BatchNorm(layers.BatchNormalization):
    """Overload for data_format understanding"""

    def __init__(self, data_format=None, momentum=0.99, epsilon=1e-3, center=True, scale=True, beta_initializer='zeros',
                 gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones',
                 beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None,
                 renorm=False, renorm_clipping=None, renorm_momentum=0.99, fused=None, trainable=True,
                 virtual_batch_size=None, adjustment=None, name=None, **kwargs):
        kwargs['autocast'] = False
        kwargs['dtype'] = 'float32'
        self.data_format = normalize_data_format(data_format)
        axis = -1 if 'channels_last' == self.data_format else 1
        super().__init__(axis=axis, momentum=momentum, epsilon=epsilon, center=center, scale=scale,
                         beta_initializer=beta_initializer, gamma_initializer=gamma_initializer,
                         moving_mean_initializer=moving_mean_initializer,
                         moving_variance_initializer=moving_variance_initializer, beta_regularizer=beta_regularizer,
                         gamma_regularizer=gamma_regularizer, beta_constraint=beta_constraint,
                         gamma_constraint=gamma_constraint, renorm=renorm, renorm_clipping=renorm_clipping,
                         renorm_momentum=renorm_momentum, fused=fused, trainable=trainable,
                         virtual_batch_size=virtual_batch_size, adjustment=adjustment, name=name, **kwargs)

    def call(self, inputs, *args, **kwargs):
        outputs = tf.cast(inputs, 'float32')
        outputs = super().call(outputs)
        outputs = tf.saturate_cast(outputs, inputs.dtype)

        return outputs

    def compute_output_signature(self, input_signature):
        return input_signature

    def get_config(self):
        config = super().get_config()
        config.update({'data_format': self.data_format})

        del config['axis']

        return config


@NORMALIZATIONS.register('ln')
@register_keras_serializable(package='SegMe>Policy>Normalization')
class LayerNorm(layers.LayerNormalization):
    """Overload casting for stability and fused implementation"""

    def __init__(self, data_format=None, epsilon=1e-3, center=True, scale=True, beta_initializer='zeros',
                 gamma_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
                 gamma_constraint=None, **kwargs):
        kwargs['autocast'] = False
        kwargs['dtype'] = 'float32'
        self.data_format = normalize_data_format(data_format)
        axis = -1 if 'channels_last' == self.data_format else 1
        super().__init__(axis=axis, epsilon=epsilon, center=center, scale=scale, beta_initializer=beta_initializer,
                         gamma_initializer=gamma_initializer, beta_regularizer=beta_regularizer,
                         gamma_regularizer=gamma_regularizer, beta_constraint=beta_constraint,
                         gamma_constraint=gamma_constraint, **kwargs)

    @shape_type_conversion
    def build(self, input_shape):
        super().build(input_shape)
        if not self._fused:
            warnings.warn(f'Layer {self.name} will use an inefficient implementation.')

    def call(self, inputs, *args, **kwargs):
        outputs = tf.cast(inputs, 'float32')
        outputs = super().call(outputs)
        outputs = tf.saturate_cast(outputs, inputs.dtype)

        return outputs

    def compute_output_signature(self, input_signature):
        return input_signature

    def get_config(self):
        config = super().get_config()
        config.update({'data_format': self.data_format})

        del config['axis']

        return config


@NORMALIZATIONS.register('gn')
@register_keras_serializable(package='SegMe>Policy>Normalization')
class GroupNorm(layers.GroupNormalization):
    """Overload casting for stability and groups reduction"""

    def __init__(self, groups=None, data_format=None, epsilon=1e-3, center=True, scale=True, beta_initializer='zeros',
                 gamma_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
                 gamma_constraint=None, **kwargs):
        kwargs['autocast'] = False
        kwargs['dtype'] = 'float32'
        self.data_format = normalize_data_format(data_format)
        axis = -1 if 'channels_last' == self.data_format else 1
        super().__init__(groups=32, axis=axis, epsilon=epsilon, center=center, scale=scale,
                         beta_initializer=beta_initializer, gamma_initializer=gamma_initializer,
                         beta_regularizer=beta_regularizer, gamma_regularizer=gamma_regularizer,
                         beta_constraint=beta_constraint, gamma_constraint=gamma_constraint, **kwargs)

        self._groups = groups

    @shape_type_conversion
    def build(self, input_shape):
        channels = input_shape[-1]
        if channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')

        if self._groups is None:
            # Best results in paper obtained with groups=32 or channels_per_group=16
            best_cpg = channels // (2 ** math.floor(math.log2(channels) / 2))
            best_grp = min(32, channels // min(16, best_cpg))
            self.groups = best_grp

        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        outputs = tf.cast(inputs, 'float32')
        outputs = super().call(outputs)
        outputs = tf.saturate_cast(outputs, inputs.dtype)

        return outputs

    def compute_output_signature(self, input_signature):
        return input_signature

    def get_config(self):
        config = super().get_config()
        config.update({
            'groups': self._groups,
            'data_format': self.data_format
        })

        del config['axis']

        return config


@NORMALIZATIONS.register('frn')
@register_keras_serializable(package='SegMe>Policy>Normalization')
class FilterResponseNorm(add_layers.FilterResponseNormalization):
    """Overload casting for stability"""

    def __init__(self, data_format=None, epsilon=1e-6, beta_initializer='zeros', gamma_initializer='ones',
                 beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None,
                 learned_epsilon=False, learned_epsilon_constraint=None, **kwargs):
        kwargs['autocast'] = False
        kwargs['dtype'] = 'float32'
        self.data_format = normalize_data_format(data_format)
        axis = [1, 2] if 'channels_last' == self.data_format else [2, 3]
        super().__init__(epsilon=epsilon, axis=axis, beta_initializer=beta_initializer,
                         gamma_initializer=gamma_initializer, beta_regularizer=beta_regularizer,
                         gamma_regularizer=gamma_regularizer, beta_constraint=beta_constraint,
                         gamma_constraint=gamma_constraint, learned_epsilon=learned_epsilon,
                         learned_epsilon_constraint=learned_epsilon_constraint, **kwargs)

    def call(self, inputs, *args, **kwargs):
        outputs = tf.cast(inputs, 'float32')
        outputs = super().call(outputs)
        outputs = tf.saturate_cast(outputs, inputs.dtype)

        return outputs

    def _check_axis(self, axis):
        if not isinstance(axis, (list, tuple)):
            raise TypeError(f'Expected a list of values but got {axis}.')
        else:
            self.axis = axis

        if 2 != len(self.axis):
            raise ValueError(
                'FilterResponseNormalization operates on per-channel basis. '
                'Axis values should be a list of spatial dimensions.'
            )

    def compute_output_signature(self, input_signature):
        return input_signature

    def get_config(self):
        config = super().get_config()
        config.update({'data_format': self.data_format})

        del config['axis']

        return config
