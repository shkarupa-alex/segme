import copy
from keras import layers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from segme.policy import cnapol


@register_keras_serializable(package='SegMe>Common')
class Conv(layers.Layer):
    def __init__(self, filters, kernel_size, conv_type=True, conv_kwargs=None, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        self.filters = filters
        self.kernel_size = kernel_size

        self.conv_kwargs = conv_kwargs
        if self.conv_kwargs and not isinstance(self.conv_kwargs, dict):
            raise ValueError('Convolution kwargs must be a dict if provided')

        policy = cnapol.global_policy()
        self.conv_type = False
        if kernel_size and isinstance(conv_type, str) and conv_type:
            self.conv_type = conv_type
        elif kernel_size and conv_type is True:
            self.conv_type = policy.conv_type
        elif kernel_size:
            raise ValueError('Kernel size should be 0, None or False if convolution is omitted')
        elif bool(conv_type):
            raise ValueError('Unknown convolution type')

    @shape_type_conversion
    def build(self, input_shape):
        if not self.conv_type:
            raise ValueError('Empty convolution layer can\'t be executed')

        conv_kwargs = self.conv_kwargs or {}
        if self.filters:
            self.conv = cnapol.SAMECONVS.new(self.conv_type, self.filters, self.kernel_size, **conv_kwargs)
        else:  # depthwise
            self.conv = cnapol.SAMECONVS.new(f'dw{self.conv_type}', self.kernel_size, **conv_kwargs)

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
            'conv_type': self.conv_type,
            'conv_kwargs': self.conv_kwargs
        })

        return config


@register_keras_serializable(package='SegMe>Common')
class Norm(layers.Layer):
    def __init__(self, norm_type=True, norm_kwargs=None, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        self.norm_kwargs = norm_kwargs
        if self.norm_kwargs and not isinstance(self.norm_kwargs, dict):
            raise ValueError('Normalization kwargs must be a dict if provided')

        policy = cnapol.global_policy()
        self.norm_type = False
        if isinstance(norm_type, str) and norm_type:
            self.norm_type = norm_type
        elif norm_type is True:
            self.norm_type = policy.norm_type
        elif bool(norm_type):
            raise ValueError('Unknown normalization type')

    @shape_type_conversion
    def build(self, input_shape):
        if not self.norm_type:
            raise ValueError('Empty normalization layer can\'t be executed')

        norm_kwargs = self.norm_kwargs or {}
        self.norm = cnapol.NORMALIZATIONS.new(self.norm_type, **norm_kwargs)
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
            'norm_type': self.norm_type,
            'norm_kwargs': self.norm_kwargs
        })

        return config


@register_keras_serializable(package='SegMe>Common')
class Act(layers.Layer):
    def __init__(self, act_type=True, act_kwargs=None, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        self.act_kwargs = act_kwargs
        if self.act_kwargs and not isinstance(self.act_kwargs, dict):
            raise ValueError('Activation kwargs must be a dict if provided')

        policy = cnapol.global_policy()
        self.act_type = False
        if isinstance(act_type, str) and act_type:
            self.act_type = act_type
        elif act_type is True:
            self.act_type = policy.act_type
        elif bool(act_type):
            raise ValueError('Unknown activation type')

    @shape_type_conversion
    def build(self, input_shape):
        if not self.act_type:
            raise ValueError('Empty activation layer can\'t be executed')

        act_kwargs = self.act_kwargs or {}
        self.act = cnapol.ACTIVATIONS.new(self.act_type, **act_kwargs)
        self.act.build(input_shape)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        return self.act(inputs)

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            'act_type': self.act_type,
            'act_kwargs': self.act_kwargs
        })

        return config


@register_keras_serializable(package='SegMe>Common')
class ConvNormAct(layers.Layer):
    def __init__(self, filters, kernel_size, conv_type=True, conv_kwargs=None, norm_type=True, norm_kwargs=None,
                 act_type=True, act_kwargs=None, cna_policy=None, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        self.filters = filters
        self.kernel_size = kernel_size
        self.conv_type = conv_type
        self.conv_kwargs = conv_kwargs
        self.norm_type = norm_type
        self.norm_kwargs = norm_kwargs
        self.act_type = act_type
        self.act_kwargs = act_kwargs
        self.cna_policy = cnapol.deserialize(cna_policy or cnapol.global_policy())

    @shape_type_conversion
    def build(self, input_shape):
        with cnapol.policy_scope(self.cna_policy):
            self.norm = Norm(self.norm_type, self.norm_kwargs)
            self.act = Act(self.act_type, self.act_kwargs)

            conv_kwargs = copy.deepcopy(self.conv_kwargs or {})
            if self.norm.norm_type:
                conv_kwargs['use_bias'] = False
            if 'relu' == self.act.act_type and not self.norm.norm_type:
                conv_kwargs['kernel_initializer'] = 'he_normal'
            self.conv = Conv(self.filters, self.kernel_size, self.conv_type, conv_kwargs)

        if not self.conv.conv_type and not self.norm.norm_type and not self.act.act_type:
            raise ValueError('All components of Conv-Norm-Act are empty')


        current_shape = input_shape
        if self.conv.conv_type:
            self.conv.build(current_shape)
            current_shape = self.conv.compute_output_shape(current_shape)
        if self.norm.norm_type:
            self.norm.build(current_shape)
        if self.act.act_type:
            self.act.build(current_shape)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = inputs

        if self.conv.conv_type:
            outputs = self.conv(outputs)
        if self.norm.norm_type:
            outputs = self.norm(outputs)
        if self.act.act_type:
            outputs = self.act(outputs)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        if not self.conv.conv_type:
            return input_shape

        return self.conv.compute_output_shape(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'conv_type': self.conv_type,
            'conv_kwargs': self.conv_kwargs,
            'norm_type': self.norm_type,
            'norm_kwargs': self.norm_kwargs,
            'act_type': self.act_type,
            'act_kwargs': self.act_kwargs,
            'cna_policy': cnapol.serialize(self.cna_policy)
        })

        return config
