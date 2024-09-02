from keras.src import layers
from keras.src import ops
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
        self.resize = BilinearInterpolation(dtype=self.dtype_policy)

        self.conv1 = Sequence(
            [
                Norm(dtype=self.dtype_policy),
                Act(dtype=self.dtype_policy),
                ConvNormAct(self.filters, 3, dtype=self.dtype_policy),
                Conv(self.filters, 3, dtype=self.dtype_policy),
            ],
            dtype=self.dtype_policy,
        )
        self.conv1.build(
            input_shape[1][:-1] + (input_shape[0][-1] + input_shape[1][-1],)
        )

        self.shortcut = Conv(self.filters, 1, dtype=self.dtype_policy)
        self.shortcut.build(input_shape[1][:-1] + (input_shape[0][-1],))

        self.conv2 = Sequence(
            [
                Norm(dtype=self.dtype_policy),
                Act(dtype=self.dtype_policy),
                ConvNormAct(self.filters, 3, dtype=self.dtype_policy),
                Conv(self.filters, 3, dtype=self.dtype_policy),
            ],
            dtype=self.dtype_policy,
        )
        self.conv2.build(input_shape[1][:-1] + (self.filters,))

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        high, low = inputs

        high = self.resize([high, low])
        outputs = self.conv1(ops.concatenate([high, low], axis=-1))
        outputs += self.shortcut(high)
        outputs += self.conv2(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[1][:-1] + (self.filters,)

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})

        return config
