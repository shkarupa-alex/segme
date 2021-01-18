from tensorflow.keras import Sequential, layers, utils
from tensorflow.python.keras.utils.tf_utils import shape_type_conversion
from ...common import resize_by_sample, ConvBnRelu


@utils.register_keras_serializable(package='SegMe>F3Net')
class CFM(layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [layers.InputSpec(ndim=4), layers.InputSpec(ndim=4)]
        self.filters = filters

    @shape_type_conversion
    def build(self, input_shape):
        self.cbr1h = ConvBnRelu(self.filters, 3, kernel_initializer='he_normal')
        self.cbr2h = ConvBnRelu(self.filters, 3, kernel_initializer='he_normal')
        self.cbr3h = ConvBnRelu(self.filters, 3, kernel_initializer='he_normal')
        self.cbr4h = ConvBnRelu(self.filters, 3, kernel_initializer='he_normal')

        self.cbr1v = ConvBnRelu(self.filters, 3, kernel_initializer='he_normal')
        self.cbr2v = ConvBnRelu(self.filters, 3, kernel_initializer='he_normal')
        self.cbr3v = ConvBnRelu(self.filters, 3, kernel_initializer='he_normal')
        self.cbr4v = ConvBnRelu(self.filters, 3, kernel_initializer='he_normal')

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        left, down = inputs
        down = resize_by_sample([down, left])

        out1h = self.cbr1h(left)
        out2h = self.cbr2h(out1h)
        out1v = self.cbr1v(down)
        out2v = self.cbr2v(out1v)
        fuse = layers.multiply([out2h, out2v])
        out3h = layers.add([self.cbr3h(fuse), out1h])
        out4h = self.cbr4h(out3h)
        out3v = layers.add([self.cbr3v(fuse), out1v])
        out4v = self.cbr4v(out3v)

        return out4h, out4v

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return [input_shape[0][:-1] + (self.filters,)] * 2

    def get_config(self):
        config = super().get_config()
        config.update({'filters': self.filters})

        return config
