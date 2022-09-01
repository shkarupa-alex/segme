import copy
from keras import activations, constraints, initializers, layers, regularizers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from segme.policy import cnapol


@register_keras_serializable(package='SegMe>Common>ConvNormAct')
class Conv(layers.Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1), data_format=None, dilation_rate=(1, 1), activation=None,
                 use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None,
                 bias_regularizer=None, kernel_constraint=None, bias_constraint=None,
                 policy=None, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.policy = cnapol.deserialize(policy or cnapol.global_policy())

    @shape_type_conversion
    def build(self, input_shape):
        conv_kwargs = {
            'kernel_size': self.kernel_size, 'strides': self.strides, 'data_format': self.data_format,
            'dilation_rate': self.dilation_rate, 'activation': self.activation, 'use_bias': self.use_bias,
            'kernel_initializer': self.kernel_initializer, 'bias_initializer': self.bias_initializer,
            'kernel_regularizer': self.kernel_regularizer, 'bias_regularizer': self.bias_regularizer,
            'kernel_constraint': self.kernel_constraint, 'bias_constraint': self.bias_constraint,
            'name': 'wrapped'}
        if self.filters:
            self.conv = cnapol.SAMECONVS.new(self.policy.conv_type, filters=self.filters, **conv_kwargs)
        else:  # depthwise
            self.conv = cnapol.SAMECONVS.new('dwconv', **conv_kwargs)
        self.conv.build(input_shape)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        return self.conv(inputs)

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return self.conv.compute_output_shape(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'policy': cnapol.serialize(self.policy)
        })

        return config


@register_keras_serializable(package='SegMe>Common>ConvNormAct')
class Norm(layers.Layer):
    def __init__(self, epsilon=None, policy=None, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        self.epsilon = epsilon
        self.policy = cnapol.deserialize(policy or cnapol.global_policy())

    @shape_type_conversion
    def build(self, input_shape):
        norm_kwargs = {'name': 'wrapped'}
        if self.epsilon:
            norm_kwargs['epsilon'] = self.epsilon

        self.norm = cnapol.NORMALIZATIONS.new(self.policy.norm_type, **norm_kwargs)
        self.norm.build(input_shape)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        return self.norm(inputs)

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            'epsilon': self.epsilon,
            'policy': cnapol.serialize(self.policy)
        })

        return config


@register_keras_serializable(package='SegMe>Common>ConvNormAct')
class Act(layers.Layer):
    def __init__(self, policy=None, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        self.policy = cnapol.deserialize(policy or cnapol.global_policy())

    @shape_type_conversion
    def build(self, input_shape):
        self.act = cnapol.ACTIVATIONS.new(self.policy.act_type, name='wrapped')
        self.act.build(input_shape)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        return self.act(inputs)

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({'policy': cnapol.serialize(self.policy)})

        return config


@register_keras_serializable(package='SegMe>Common>ConvNormAct')
class ConvAct(layers.Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1), data_format=None, dilation_rate=(1, 1), use_bias=True,
                 kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None,
                 bias_regularizer=None, kernel_constraint=None, bias_constraint=None, policy=None, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.policy = cnapol.deserialize(policy or cnapol.global_policy())

    @shape_type_conversion
    def build(self, input_shape):
        if 'relu' == self.policy.act_type and 'glorot_uniform' == initializers.serialize(self.kernel_initializer):
            kernel_initializer = 'he_normal'
        else:
            kernel_initializer = self.kernel_initializer

        self.conv = Conv(
            filters=self.filters, kernel_size=self.kernel_size, strides=self.strides, data_format=self.data_format,
            dilation_rate=self.dilation_rate, use_bias=self.use_bias, kernel_initializer=kernel_initializer,
            bias_initializer=self.bias_initializer, kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer, kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint, policy=self.policy, name='policy_conv')
        self.act = Act(policy=self.policy, name='policy_act')

        current_shape = input_shape
        self.conv.build(current_shape)

        current_shape = self.conv.compute_output_shape(current_shape)
        self.act.build(current_shape)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = self.conv(inputs)
        outputs = self.act(outputs)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return self.conv.compute_output_shape(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'policy': cnapol.serialize(self.policy)
        })

        return config


@register_keras_serializable(package='SegMe>Common>ConvNormAct')
class ConvNormAct(layers.Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1), data_format=None, dilation_rate=(1, 1), use_bias=False,
                 kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None,
                 bias_regularizer=None, kernel_constraint=None, bias_constraint=None, epsilon=None, policy=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.epsilon = epsilon
        self.policy = cnapol.deserialize(policy or cnapol.global_policy())

    @shape_type_conversion
    def build(self, input_shape):
        self.conv = Conv(
            filters=self.filters, kernel_size=self.kernel_size, strides=self.strides, data_format=self.data_format,
            dilation_rate=self.dilation_rate, use_bias=self.use_bias, kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer, kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer, kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint, policy=self.policy, name='policy_conv')
        self.norm = Norm(epsilon=self.epsilon, policy=self.policy, name='policy_norm')
        self.act = Act(policy=self.policy, name='policy_act')

        current_shape = input_shape
        self.conv.build(current_shape)

        current_shape = self.conv.compute_output_shape(current_shape)
        self.norm.build(current_shape)
        self.act.build(current_shape)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = self.conv(inputs)
        outputs = self.norm(outputs)
        outputs = self.act(outputs)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return self.conv.compute_output_shape(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'epsilon': self.epsilon,
            'policy': cnapol.serialize(self.policy)
        })

        return config
