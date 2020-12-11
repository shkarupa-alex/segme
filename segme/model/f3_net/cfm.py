from tensorflow.keras import Sequential, layers, utils
from tensorflow.python.keras.utils.tf_utils import shape_type_conversion
from ...common import resize_by_sample


@utils.register_keras_serializable(package='SegMe>F3Net')
class CFM(layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [layers.InputSpec(ndim=4), layers.InputSpec(ndim=4)]
        self.filters = filters

    @shape_type_conversion
    def build(self, input_shape):
        self.cbr1h = Sequential([
            layers.Conv2D(self.filters, 3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ])
        self.cbr2h = Sequential([
            layers.Conv2D(self.filters, 3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()])
        self.cbr3h = Sequential([
            layers.Conv2D(self.filters, 3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()])
        self.cbr4h = Sequential([
            layers.Conv2D(self.filters, 3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()])

        self.cbr1v = Sequential([
            layers.Conv2D(self.filters, 3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()])
        self.cbr2v = Sequential([
            layers.Conv2D(self.filters, 3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()])
        self.cbr3v = Sequential([
            layers.Conv2D(self.filters, 3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()])
        self.cbr4v = Sequential([
            layers.Conv2D(self.filters, 3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()])

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
