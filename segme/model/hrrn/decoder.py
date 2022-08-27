import tensorflow as tf
from keras import activations, initializers, layers, models
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from tensorflow_addons.layers import SpectralNormalization
from ...common import ResizeByScale


@register_keras_serializable(package='SegMe>HRRN')
class Decoder(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [layers.InputSpec(ndim=4) for _ in range(6)]

    def _make_layer(self, filters, num_repeats, bn_epsilon=1e-5, bn_momentum=0.0, activation='leaky_relu'):
        group = []
        for i in range(num_repeats):
            is_first = 0 == i
            group.append(Bottleneck(
                filters=filters,
                strides=2 if is_first else 1,  # TODO: last?
                use_projection=is_first,
                activation=activation,
                bn_epsilon=bn_epsilon,
                bn_momentum=bn_momentum))

        return models.Sequential(group)

    @shape_type_conversion
    def build(self, input_shape):
        filters = [shape[-1] for shape in input_shape[:-1]]

        self.layer16 = self._make_layer(filters[4], 3)
        self.layer8 = self._make_layer(filters[3], 6)
        self.layer4 = self._make_layer(filters[2], 4)
        self.layer2 = self._make_layer(filters[1], 3)
        self.layer1 = models.Sequential([
            SpectralNormalization(layers.Conv2DTranspose(filters[0], 4, strides=2, padding='same', use_bias=False)),
            layers.BatchNormalization(),
            layers.Activation('leaky_relu')
        ])

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        short1, short2, short4, short8, short16, feats32 = inputs

        outputs = self.layer16(feats32) + short16
        outputs = self.layer8(outputs) + short8
        outputs = self.layer4(outputs) + short4
        outputs = self.layer2(outputs) + short2
        outputs = self.layer1(outputs) + short1

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[0]


@register_keras_serializable(package='SegMe>HRRN')
class Bottleneck(layers.Layer):
    def __init__(self, filters, strides, use_projection, bn_momentum, bn_epsilon, activation, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        self.filters = filters
        self.strides = strides
        self.use_projection = use_projection
        self.bn_momentum = bn_momentum
        self.bn_epsilon = bn_epsilon
        self.activation = activations.get(activation)

    @shape_type_conversion
    def build(self, input_shape):
        self.channels = input_shape[-1]
        if self.channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')
        self.input_spec = layers.InputSpec(ndim=4, axes={-1: self.channels})

        if self.channels != self.filters and not self.use_projection:
            raise ValueError('Channel dimension of inputs should equals to filters value if no projection used.')

        if self.use_projection:
            self.short = models.Sequential([
                SpectralNormalization(layers.Conv2D(self.filters, 1, use_bias=False)),
                layers.BatchNormalization(momentum=self.bn_momentum, epsilon=self.bn_epsilon),
                ResizeByScale(self.strides, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)  # TODO: try bilinear
            ])

        self.block = models.Sequential([
            # First conv layer
            SpectralNormalization(layers.Conv2D(self.channels, 1, use_bias=False)),  # TODO: try reduce channels first
            layers.BatchNormalization(momentum=self.bn_momentum, epsilon=self.bn_epsilon),
            layers.Activation(self.activation),

            # Second conv layer
            SpectralNormalization(
                layers.Conv2D(self.channels, 3, padding='same', use_bias=False) if 1 == self.strides
                # TODO: try bilinear
                else layers.Conv2DTranspose(self.channels, 4, strides=2, padding='same', use_bias=False)),
            layers.BatchNormalization(momentum=self.bn_momentum, epsilon=self.bn_epsilon),
            layers.Activation(self.activation),

            # Third conv layer
            SpectralNormalization(layers.Conv2D(self.filters, 1, use_bias=False)),
            layers.BatchNormalization(momentum=self.bn_momentum, epsilon=self.bn_epsilon),

            SqueezeExcitation()
        ])

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        shortcut = inputs

        if self.use_projection:
            shortcut = self.short(shortcut)

        outputs = self.block(inputs)

        outputs += shortcut
        outputs = self.activation(outputs)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.filters,)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'strides': self.strides,
            'use_projection': self.use_projection,
            'bn_momentum': self.bn_momentum,
            'bn_epsilon': self.bn_epsilon,
            'activation': activations.serialize(self.activation)
        })

        return config


@register_keras_serializable(package='SegMe>HRRN')
class SqueezeExcitation(layers.Layer):
    def __init__(self, ratio=1, **kwargs):  # TODO: ratio
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        self.ratio = ratio

    @shape_type_conversion
    def build(self, input_shape):
        self.channels = input_shape[-1]
        if self.channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')
        self.input_spec = layers.InputSpec(ndim=4, axes={-1: self.channels})

        squeeze_filters = max(1, self.channels // self.ratio)
        kernel_init = initializers.get({
            'class_name': 'VarianceScaling',
            'config': {'scale': 2.0, 'mode': 'fan_out', 'distribution': 'truncated_normal'}})

        self.se = models.Sequential([
            layers.GlobalAvgPool2D(keepdims=True),
            layers.Conv2D(squeeze_filters, 1, kernel_initializer=kernel_init, activation='relu'),
            layers.Conv2D(self.channels, 1, kernel_initializer=kernel_init, activation='sigmoid')
        ])

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = self.se(inputs) * inputs
        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({'ratio': self.ratio})

        return config
