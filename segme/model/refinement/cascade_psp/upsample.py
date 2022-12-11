import tensorflow as tf
from keras import layers
from keras.saving.object_registration import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from segme.common.convnormact import ConvNormAct, Conv, Norm, Act
from segme.common.interrough import BilinearInterpolation
from segme.common.sequent import Sequential


@register_keras_serializable(package='SegMe>Model>Refinement>CascadePSP')
class Upsample(layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [layers.InputSpec(ndim=4), layers.InputSpec(ndim=4)]
        self.filters = filters

    @shape_type_conversion
    def build(self, input_shape):
        self.resize = BilinearInterpolation(None)
        self.conv1 = Sequential([
            Norm(),
            Act(),
            ConvNormAct(self.filters, 3),
            Conv(self.filters, 3)
        ])
        self.conv2 = Sequential([
            Norm(),
            Act(),
            ConvNormAct(self.filters, 3),
            Conv(self.filters, 3)
        ])
        self.shortcut = Conv(self.filters, 1)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        high, low = inputs

        high = self.resize([high, low])
        outputs = self.conv1(tf.concat([high, low], axis=-1))
        outputs += self.shortcut(high)
        outputs += self.conv2(outputs)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[1][:-1] + (self.filters,)

    def get_config(self):
        config = super().get_config()
        config.update({'filters': self.filters})

        return config
