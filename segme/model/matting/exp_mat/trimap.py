from keras.src import KerasTensor
from keras.src import layers
from keras.src import ops
from keras.src.layers.input_spec import InputSpec
from keras.src.saving import register_keras_serializable


@register_keras_serializable(package="SegMe>Model>Matting>ExpMat")
class Trimap(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(
            ndim=4, axes={-1: 1}, dtype="uint8"
        )  # trimap

    def call(self, inputs, **kwargs):
        trimap = ops.squeeze(inputs // 86, -1)
        trimap = ops.one_hot(trimap, 3, dtype="int32")
        trimap = ops.clip(trimap * 255, 0, 255)
        trimap = ops.cast(trimap, "uint8")

        return trimap

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (3,)

    def compute_output_spec(self, inputs, training=False):
        output_spec = super().compute_output_spec(inputs, training=training)

        return KerasTensor(output_spec.shape, dtype=inputs.dtype)
