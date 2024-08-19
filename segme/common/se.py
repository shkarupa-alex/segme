from keras.src import layers
from keras.src.layers.input_spec import InputSpec
from keras.src.saving import register_keras_serializable

from segme.common.convnormact import ConvAct
from segme.common.sequence import Sequence


@register_keras_serializable(package="SegMe>Common")
class SE(layers.Layer):
    def __init__(self, ratio=0.25, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=4)

        if not 0.0 <= ratio <= 1.0:
            raise ValueError("Squeeze ratio must be in range [0; 1].")

        self.ratio = ratio

    def build(self, input_shape):
        channels = input_shape[-1]
        if channels is None:
            raise ValueError(
                "Channel dimension of the inputs should be defined. "
                "Found `None`."
            )

        filters = max(1, int(channels * self.ratio))
        self.se = Sequence(
            [
                layers.GlobalAveragePooling2D(
                    keepdims=True, name="pool", dtype=self.dtype_policy
                ),
                ConvAct(
                    filters,
                    1,
                    kernel_initializer="variance_scaling",
                    name="fc0",
                    dtype=self.dtype_policy,
                ),
                layers.Conv2D(
                    channels,
                    1,
                    activation="sigmoid",
                    kernel_initializer="variance_scaling",
                    name="fc1",
                    dtype=self.dtype_policy,
                ),
            ],
            name="se",
            dtype=self.dtype_policy,
        )
        self.se.build(input_shape)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = inputs * self.se(inputs)

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({"ratio": self.ratio})

        return config
