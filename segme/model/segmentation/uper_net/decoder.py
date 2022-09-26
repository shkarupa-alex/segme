import tensorflow as tf
from keras import layers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from segme.common.convnormact import ConvNormAct
from segme.common.ppm import PyramidPooling
from segme.common.interrough import BilinearInterpolation


@register_keras_serializable(package='SegMe>Model>Segmentation>UPerNet')
class Decoder(layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters

    @shape_type_conversion
    def build(self, input_shape):
        if not all([isinstance(s, (tuple, list)) for s in input_shape]):
            raise ValueError('Wrong inputs count')
        self.scales = len(input_shape)
        self.input_spec = [layers.InputSpec(ndim=4) for _ in range(self.scales)]

        self.resize = BilinearInterpolation(None)
        self.psp = PyramidPooling(self.filters)
        self.lat_convs = [ConvNormAct(self.filters, 1) for _ in range(self.scales - 1)]
        self.fpn_convs = [ConvNormAct(self.filters, 3) for _ in range(self.scales - 1)]
        self.bottleneck = ConvNormAct(self.filters, 3)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        laterals = [self.lat_convs[i](inputs[i]) for i in range(self.scales - 1)]
        laterals.append(self.psp(inputs[-1]))

        for i in range(self.scales - 1, 0, -1):
            laterals[i - 1] += self.resize([laterals[i], laterals[i - 1]])

        outputs = [self.fpn_convs[i](laterals[i]) for i in range(self.scales - 1)]
        outputs.append(laterals[-1])

        for i in range(self.scales - 1, 0, -1):
            outputs[i] = self.resize([outputs[i], outputs[0]])

        outputs = tf.concat(outputs, axis=-1)
        outputs = self.bottleneck(outputs)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[0][:-1] + (self.filters,)

    def get_config(self):
        config = super().get_config()
        config.update({'filters': self.filters})

        return config
