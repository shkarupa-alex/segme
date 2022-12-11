import tensorflow as tf
from keras import layers
from keras.saving.object_registration import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from segme.common.convnormact import ConvNormAct, Conv, Norm, Act
from segme.common.interrough import NearestInterpolation


@register_keras_serializable(package='SegMe>Model>SOD>MINet')
class Conv3nV1(layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [layers.InputSpec(ndim=4), layers.InputSpec(ndim=4), layers.InputSpec(ndim=4)]
        self.filters = filters

    @shape_type_conversion
    def build(self, input_shape):
        self.channels_h = input_shape[0][-1]
        self.channels_m = input_shape[1][-1]
        self.channels_l = input_shape[2][-1]
        if self.channels_h is None or self.channels_m is None or self.channels_l is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')

        self.input_spec = [
            layers.InputSpec(ndim=4, axes={-1: self.channels_h}),
            layers.InputSpec(ndim=4, axes={-1: self.channels_m}),
            layers.InputSpec(ndim=4, axes={-1: self.channels_l})
        ]

        min_channels = min(self.channels_h, self.channels_m, self.channels_l)

        self.resize = NearestInterpolation(None)
        self.pool = layers.AveragePooling2D(2, strides=2, padding='same')
        self.act = Act()

        # stage 0
        self.cna_hh0 = ConvNormAct(min_channels, 3)
        self.cna_mm0 = ConvNormAct(min_channels, 3)
        self.cna_ll0 = ConvNormAct(min_channels, 3)

        # stage 1
        self.conv_hh1 = Conv(min_channels, 3)
        self.conv_hm1 = Conv(min_channels, 3)
        self.conv_mh1 = Conv(min_channels, 3)
        self.conv_mm1 = Conv(min_channels, 3)
        self.conv_ml1 = Conv(min_channels, 3)
        self.conv_lm1 = Conv(min_channels, 3)
        self.conv_ll1 = Conv(min_channels, 3)
        self.norm_h1 = Norm()
        self.norm_m1 = Norm()
        self.norm_l1 = Norm()

        # stage 2
        self.conv_hm2 = Conv(min_channels, 3)
        self.conv_lm2 = Conv(min_channels, 3)
        self.conv_mm2 = Conv(min_channels, 3)
        self.norm_m2 = Norm()

        # stage 3
        self.conv_mm3 = Conv(self.filters, 3)
        self.norm_m3 = Norm()

        self.identity = Conv(self.filters, 1)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        inputs_h, inputs_m, inputs_l = inputs

        # stage 0
        h = self.cna_hh0(inputs_h)
        m = self.cna_mm0(inputs_m)
        l = self.cna_ll0(inputs_l)

        # stage 1
        h2h = self.conv_hh1(h)
        m2h = self.conv_mh1(self.resize([m, h2h]))

        h2m = self.conv_hm1(self.pool(h))
        m2m = self.conv_mm1(m)
        l2m = self.conv_lm1(self.resize([l, m2m]))

        m2l = self.conv_ml1(self.pool(m))
        l2l = self.conv_ll1(l)

        h = self.act(self.norm_h1(h2h + m2h))
        m = self.act(self.norm_m1(h2m + m2m + l2m))
        l = self.act(self.norm_l1(m2l + l2l))

        # stage 2
        h2m = self.conv_hm2(self.pool(h))
        m2m = self.conv_mm2(m)
        l2m = self.conv_lm2(self.resize([l, m2m]))
        m = self.act(self.norm_m2(h2m + m2m + l2m))

        # stage 3
        out = self.norm_m3(self.conv_mm3(m)) + self.identity(inputs_m)

        return self.act(out)

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[1][:-1] + (self.filters,)

    def get_config(self):
        config = super().get_config()
        config.update({'filters': self.filters})

        return config
