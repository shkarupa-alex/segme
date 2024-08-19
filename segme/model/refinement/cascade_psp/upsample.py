import tensorflow as tf
from keras.src import layers
from keras.src.layers.input_spec import InputSpec
from keras.src.saving import register_keras_serializable

from segme.common.convnormact import Act
from segme.common.convnormact import Conv
from segme.common.convnormact import ConvNormAct
from segme.common.convnormact import Norm
from segme.common.resize import BilinearInterpolation
from segme.common.sequence import Sequence


@register_keras_serializable(package="SegMe>Model>Refinement>CascadePSP")
class Upsample(layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [InputSpec(ndim=4), InputSpec(ndim=4)]
        self.filters = filters

    def build(self, input_shape):
        self.resize = BilinearInterpolation(None)
        self.conv1 = Sequence(
            [Norm(), Act(), ConvNormAct(self.filters, 3), Conv(self.filters, 3)]
        )
        self.conv2 = Sequence(
            [Norm(), Act(), ConvNormAct(self.filters, 3), Conv(self.filters, 3)]
        )
        self.shortcut = Conv(self.filters, 1)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        high, low = inputs

        high = self.resize([high, low])
        outputs = self.conv1(tf.concat([high, low], axis=-1))
        outputs += self.shortcut(high)
        outputs += self.conv2(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[1][:-1] + (self.filters,)

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})

        return config
