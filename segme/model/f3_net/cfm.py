from keras import layers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from ...common import resize_by_sample, ConvNormRelu


@register_keras_serializable(package='SegMe>F3Net')
class CFM(layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [layers.InputSpec(ndim=4), layers.InputSpec(ndim=4)]
        self.filters = filters

    @shape_type_conversion
    def build(self, input_shape):
        self.cbr1h = ConvNormRelu(self.filters, 3, kernel_initializer='he_normal')
        self.cbr2h = ConvNormRelu(self.filters, 3, kernel_initializer='he_normal')
        self.cbr3h = ConvNormRelu(self.filters, 3, kernel_initializer='he_normal')
        self.cbr4h = ConvNormRelu(self.filters, 3, kernel_initializer='he_normal')

        self.cbr1v = ConvNormRelu(self.filters, 3, kernel_initializer='he_normal')
        self.cbr2v = ConvNormRelu(self.filters, 3, kernel_initializer='he_normal')
        self.cbr3v = ConvNormRelu(self.filters, 3, kernel_initializer='he_normal')
        self.cbr4v = ConvNormRelu(self.filters, 3, kernel_initializer='he_normal')

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        left, down = inputs
        down = resize_by_sample([down, left])

        out1h = self.cbr1h(left)
        out2h = self.cbr2h(out1h)
        out1v = self.cbr1v(down)
        out2v = self.cbr2v(out1v)
        fuse = out2h + out2v
        out3h = self.cbr3h(fuse) + out1h
        out4h = self.cbr4h(out3h)
        out3v = self.cbr3v(fuse) + out1v
        out4v = self.cbr4v(out3v)

        return out4h, out4v

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return [input_shape[0][:-1] + (self.filters,)] * 2

    def get_config(self):
        config = super().get_config()
        config.update({'filters': self.filters})

        return config
