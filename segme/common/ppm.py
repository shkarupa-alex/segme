from keras.src import layers
from keras.src import ops
from keras.src.layers.input_spec import InputSpec
from keras.src.saving import register_keras_serializable

from segme.common.convnormact import ConvNormAct
from segme.common.pool import AdaptiveAveragePooling
from segme.common.resize import BilinearInterpolation
from segme.common.sequence import Sequence


@register_keras_serializable(package="SegMe>Common")
class PyramidPooling(layers.Layer):
    def __init__(self, filters, sizes=(1, 2, 3, 6), **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=4)

        self.filters = filters
        self.sizes = sizes

    def build(self, input_shape):
        self.stages = []
        for size in self.sizes:
            s = Sequence(
                [
                    AdaptiveAveragePooling(
                        size, name="pool", dtype=self.dtype_policy
                    ),
                    ConvNormAct(
                        self.filters, 1, name="cna", dtype=self.dtype_policy
                    ),
                ],
                name=f"stage_{size}",
                dtype=self.dtype_policy,
            )
            s.build(input_shape)
            self.stages.append(s)

        self.interpolate = BilinearInterpolation(dtype=self.dtype_policy)

        self.bottleneck = ConvNormAct(
            self.filters, 3, name="bottleneck", dtype=self.dtype_policy
        )
        self.bottleneck.build(
            input_shape[:-1]
            + (input_shape[-1] + self.filters * len(self.sizes),)
        )

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = [stage(inputs) for stage in self.stages]
        outputs = [self.interpolate([output, inputs]) for output in outputs]
        outputs = ops.concatenate([inputs] + outputs, axis=-1)
        outputs = self.bottleneck(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.filters,)

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters, "sizes": self.sizes})

        return config
