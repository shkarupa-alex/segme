from keras.src import KerasTensor
from keras.src import layers
from keras.src import ops
from keras.src.layers.input_spec import InputSpec
from keras.src.saving import register_keras_serializable


@register_keras_serializable(package="SegMe>Common")
class InputGuard(layers.Layer):
    def __init__(self, **kwargs):
        kwargs.pop("autocast", None)
        super().__init__(autocast=False, **kwargs)
        self.input_spec = InputSpec(ndim=4)

    def build(self, input_shape):
        self.channels = input_shape[-1]
        if self.channels is None:
            raise ValueError(
                "Channel dimension of the inputs should be defined. "
                "Found `None`."
            )

        super().build(input_shape)

    def call(self, inputs):
        if self.channels > 3:
            return inputs[..., :3]

        if self.channels < 3:
            return ops.pad(
                inputs, ((0, 0), (0, 0), (0, 0), (0, 3 - self.channels))
            )

        return inputs

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (3,)

    def compute_output_spec(self, input_spec):
        return KerasTensor(input_spec.shape[:-1] + (3,), dtype=input_spec.dtype)
