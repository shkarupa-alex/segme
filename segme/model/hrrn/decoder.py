import tensorflow as tf
from keras import layers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from segme.common.convnormact import ConvNormAct, Conv, Norm, Act
from segme.common.interrough import NearestInterpolation
from segme.common.se import SE
from segme.common.sequent import Sequential


@register_keras_serializable(package='SegMe>Model>HRRN')
class Decoder(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [layers.InputSpec(ndim=4) for _ in range(6)]

    def _make_layer(self, filters, num_repeats):
        group = []
        for i in range(num_repeats):
            is_first = 0 == i
            group.append(Bottleneck(
                filters=filters,
                strides=2 if is_first else 1,  # TODO: last?
                use_projection=is_first))

        return Sequential(group)

    @shape_type_conversion
    def build(self, input_shape):
        filters = [shape[-1] for shape in input_shape[:-1]]

        self.layer16 = self._make_layer(filters[4], 3)
        self.layer8 = self._make_layer(filters[3], 6)
        self.layer4 = self._make_layer(filters[2], 4)
        self.layer2 = self._make_layer(filters[1], 3)
        self.layer1 = Sequential([
            layers.Conv2DTranspose(filters[0], 4, strides=2, padding='same', use_bias=False),  # TODO + SpecNorm?
            Norm(),
            Act()])

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


@register_keras_serializable(package='SegMe>Model>HRRN')
class Bottleneck(layers.Layer):
    def __init__(self, filters, strides, use_projection, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        self.filters = filters
        self.strides = strides
        self.use_projection = use_projection

    @shape_type_conversion
    def build(self, input_shape):
        self.channels = input_shape[-1]
        if self.channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')
        self.input_spec = layers.InputSpec(ndim=4, axes={-1: self.channels})

        if self.channels != self.filters and not self.use_projection:
            raise ValueError('Channel dimension of inputs should equals to filters value if no projection used.')

        if self.use_projection:
            self.short = Sequential([
                Conv(self.filters, 1, use_bias=False),
                Norm(),
                NearestInterpolation(self.strides)])

        self.block = Sequential([
            # First conv layer
            ConvNormAct(self.channels, 1),

            # Second conv layer

            Conv(self.channels, 3, use_bias=False) if 1 == self.strides
            # TODO: try bilinear
            else layers.Conv2DTranspose(self.channels, 4, strides=2, padding='same', use_bias=False),  # TODO
            Norm(),
            Act(),

            # Third conv layer
            Conv(self.filters, 1, use_bias=False),
            Norm(),

            SE()
        ])
        self.activation = Act()

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
            'use_projection': self.use_projection
        })

        return config
