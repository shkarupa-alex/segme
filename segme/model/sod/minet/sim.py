from keras import layers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from segme.common.convnormact import ConvNormAct, Conv, Norm, Act
from segme.common.interrough import BilinearInterpolation


@register_keras_serializable(package='SegMe>Model>SOD>MINet')
class SIM(layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)
        self.filters = filters

    @shape_type_conversion
    def build(self, input_shape):
        self.channels = input_shape[-1]
        if self.channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')

        self.input_spec = layers.InputSpec(ndim=4, axes={-1: self.channels})

        self.resize = BilinearInterpolation(None)
        self.pool = layers.AveragePooling2D(2, strides=2, padding='same')
        self.act = Act()

        self.cna_hh0 = ConvNormAct(self.channels, 3)
        self.cna_hl0 = ConvNormAct(self.filters, 3)

        self.conv_hh1 = Conv(self.channels, 3)
        self.conv_hl1 = Conv(self.filters, 3)
        self.conv_lh1 = Conv(self.channels, 3)
        self.conv_ll1 = Conv(self.filters, 3)
        self.norm_l1 = Norm()
        self.norm_h1 = Norm()

        self.conv_hh2 = Conv(self.channels, 3)
        self.conv_lh2 = Conv(self.channels, 3)
        self.norm_h2 = Norm()

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        # first conv
        h = self.cna_hh0(inputs)
        l = self.cna_hl0(self.pool(inputs))

        # mid conv
        h2h = self.conv_hh1(h)
        h2l = self.conv_hl1(self.pool(h))
        l2l = self.conv_ll1(l)
        l2h = self.conv_lh1(self.resize([l, inputs]))
        h = self.act(self.norm_h1(h2h + l2h))
        l = self.act(self.norm_l1(l2l + h2l))

        # last conv
        h2h = self.conv_hh2(h)
        l2h = self.conv_lh2(self.resize([l, inputs]))
        h = self.act(self.norm_h2(h2h + l2h))

        return h

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({'filters': self.filters})

        return config
