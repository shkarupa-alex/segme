import tensorflow as tf
from keras import layers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from segme.common.convnormact import ConvNormAct, Conv, Norm, Act
from segme.common.interrough import NearestInterpolation


@register_keras_serializable(package='SegMe>Model>MINet')
class Conv2nV1(layers.Layer):
    def __init__(self, filters, main, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [layers.InputSpec(ndim=4), layers.InputSpec(ndim=4)]

        if main not in {0, 1}:
            raise ValueError('Parameter "main" should equals 0 or 1')
        self.filters = filters
        self.main = main

    @shape_type_conversion
    def build(self, input_shape):
        self.channels_h = input_shape[0][-1]
        self.channels_l = input_shape[1][-1]
        if self.channels_h is None or self.channels_l is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')

        self.input_spec = [
            layers.InputSpec(ndim=4, axes={-1: self.channels_h}),
            layers.InputSpec(ndim=4, axes={-1: self.channels_l})
        ]

        min_channels = min(self.channels_h, self.channels_l)

        self.resize = NearestInterpolation(None)
        self.pool = layers.AveragePooling2D(2, strides=2, padding='same')
        self.act = Act()

        # stage 0
        self.cna_hh0 = ConvNormAct(min_channels, 3)
        self.cna_ll0 = ConvNormAct(min_channels, 3)

        # stage 1
        self.conv_hh1 = Conv(min_channels, 3)
        self.conv_hl1 = Conv(min_channels, 3)
        self.conv_lh1 = Conv(min_channels, 3)
        self.conv_ll1 = Conv(min_channels, 3)
        self.norm_l1 = Norm()
        self.norm_h1 = Norm()

        if self.main == 0:
            # stage 2
            self.conv_hh2 = Conv(min_channels, 3)
            self.conv_lh2 = Conv(min_channels, 3)
            self.norm_h2 = Norm()

            # stage 3
            self.conv_hh3 = Conv(self.filters, 3)
            self.norm_h3 = Norm()

            self.identity = Conv(self.filters, 1)

        elif self.main == 1:
            # stage 2
            self.conv_hl2 = Conv(min_channels, 3)
            self.conv_ll2 = Conv(min_channels, 3)
            self.norm_l2 = Norm()

            # stage 3
            self.conv_ll3 = Conv(self.filters, 3)
            self.norm_l3 = Norm()

            self.identity = Conv(self.filters, 1)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        inputs_h, inputs_l = inputs

        # stage 0
        h = self.cna_hh0(inputs_h)
        l = self.cna_ll0(inputs_l)

        # stage 1
        h2h = self.conv_hh1(h)
        h2l = self.conv_hl1(self.pool(h))
        l2l = self.conv_ll1(l)
        l2h = self.conv_lh1(self.resize([l, h2h]))
        h = self.act(self.norm_h1(h2h + l2h))
        l = self.act(self.norm_l1(l2l + h2l))

        if self.main == 0:
            # stage 2
            h2h = self.conv_hh2(h)
            l2h = self.conv_lh2(self.resize([l, h2h]))
            h_fuse = self.act(self.norm_h2(h2h + l2h))

            # stage 3
            out = self.norm_h3(self.conv_hh3(h_fuse)) + self.identity(inputs_h)
        else:  # self.main == 1
            # stage 2
            h2l = self.conv_hl2(self.pool(h))
            l2l = self.conv_ll2(l)
            l_fuse = self.act(self.norm_l2(h2l + l2l))

            # stage 3
            out = self.norm_l3(self.conv_ll3(l_fuse)) + self.identity(inputs_l)

        return self.act(out)

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[self.main][:-1] + (self.filters,)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'main': self.main
        })

        return config
