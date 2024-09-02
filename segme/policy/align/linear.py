from keras.src import layers
from keras.src.layers.input_spec import InputSpec
from keras.src.saving import register_keras_serializable

from segme.common.convnormact import Conv
from segme.common.resize import BilinearInterpolation


@register_keras_serializable(package="SegMe>Policy>Align")
class BilinearFeatureAlignment(layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [
            InputSpec(ndim=4),  # fine
            InputSpec(ndim=4),
        ]  # coarse

        self.filters = filters

    def build(self, input_shape):
        channels = [shape[-1] for shape in input_shape]
        if None in channels:
            raise ValueError(
                "Channel dimension of the inputs should be defined. "
                "Found `None`."
            )
        self.input_spec = [
            InputSpec(ndim=4, axes={-1: channels[0]}),
            InputSpec(ndim=4, axes={-1: channels[1]}),
        ]

        self.resize = BilinearInterpolation(dtype=self.dtype_policy)

        self.lateral = Conv(channels[0], 1, dtype=self.dtype_policy)
        self.lateral.build(input_shape[0])

        self.proj = Conv(self.filters, 3, dtype=self.dtype_policy)
        self.proj.build(input_shape[0][:-1] + (sum(channels),))

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        fine, coarse = inputs

        coarse = self.resize([coarse, fine])
        fine = self.lateral(fine)

        outputs = layers.concatenate([coarse, fine])
        outputs = self.proj(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0][:-1] + (self.filters,)

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})

        return config
