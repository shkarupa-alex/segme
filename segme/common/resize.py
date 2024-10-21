import numpy as np
from keras.src import layers
from keras.src import ops
from keras.src.layers.input_spec import InputSpec
from keras.src.saving import register_keras_serializable


@register_keras_serializable(package="SegMe>Common>Interpolation")
class NearestInterpolation(layers.Layer):
    def __init__(self, scale=None, **kwargs):
        super().__init__(**kwargs)
        if scale is not None:
            self.input_spec = InputSpec(ndim=4)
        else:
            # targets, samples
            self.input_spec = [InputSpec(ndim=4), InputSpec(ndim=4)]

        self.scale = None if scale is None else float(scale)

    def resize(self, inputs, size):
        if isinstance(self.scale, int):
            shape = ops.shape(inputs)
            outputs = inputs[:, :, None, :, None]
            outputs = ops.tile(outputs, [1, 1, self.scale, 1, self.scale, 1])
            return ops.reshape(
                outputs,
                (
                    shape[0],
                    shape[1] * self.scale,
                    shape[2] * self.scale,
                    shape[3],
                ),
            )

        return ops.image.resize(inputs, size, interpolation="nearest")

    def call(self, inputs, **kwargs):
        if 1.0 == self.scale:
            return inputs

        if self.scale is None:
            targets, samples = inputs
            new_size = ops.shape(samples)[1:3]
            static_size = all(map(lambda x: isinstance(x, int), new_size))
        else:
            targets = inputs
            new_size = ops.shape(targets)[1:3]
            static_size = all(map(lambda x: isinstance(x, int), new_size))

            if static_size:
                new_size = np.array(new_size, "float32") * self.scale
                new_size = np.round(new_size).astype("int32")
            else:
                new_size = ops.cast(new_size, self.compute_dtype) * self.scale
                new_size = ops.cast(ops.round(new_size), "int32")

        target_size = ops.shape(targets)[1:3]
        target_static = all(map(lambda x: isinstance(x, int), target_size))
        if target_static and (1, 1) == target_size:
            if static_size:
                repeats = (1,) + new_size + (1,)
            else:
                repeats = ops.concatenate([[1], new_size, [1]], axis=-1)
            outputs = ops.tile(targets, repeats)
        else:
            outputs = self.resize(targets, new_size)

        return outputs

    def compute_output_shape(self, input_shape):
        if 1 == self.scale:
            return input_shape

        if self.scale is None:
            targets_shape, samples_shape = input_shape
            return (
                (targets_shape[-0],) + samples_shape[1:3] + (targets_shape[3],)
            )

        def _scale(value):
            return None if value is None else int(round(value * self.scale))

        return (
            input_shape[0],
            _scale(input_shape[1]),
            _scale(input_shape[2]),
            input_shape[3],
        )

    def get_config(self):
        config = super().get_config()
        config.update({"scale": self.scale})

        return config


@register_keras_serializable(package="SegMe>Common>Interpolation")
class BilinearInterpolation(NearestInterpolation):
    def resize(self, inputs, size):
        return ops.image.resize(inputs, size, interpolation="bilinear")
