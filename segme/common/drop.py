import math

from keras.src import KerasTensor
from keras.src import backend
from keras.src import layers
from keras.src import ops
from keras.src.layers.input_spec import InputSpec
from keras.src.saving import register_keras_serializable


@register_keras_serializable(package="SegMe>Common")
class DropPath(layers.Dropout):
    def __init__(self, rate, seed=None, **kwargs):
        kwargs.pop("noise_shape", None)
        super().__init__(rate=rate, seed=seed, **kwargs)

    def call(self, inputs, training=False):
        if 0.0 == self.rate or not training:
            return inputs

        batch_size = ops.shape(inputs)[:1]
        noise_shape = batch_size + (1,) * (ops.ndim(inputs) - 1)

        return backend.random.dropout(
            inputs,
            self.rate,
            noise_shape=noise_shape,
            seed=self.seed_generator,
        )

    def get_config(self):
        config = super().get_config()
        del config["noise_shape"]

        return config


@register_keras_serializable(package="SegMe>Common")
class SlicePath(layers.Dropout):
    """Proposed in https://arxiv.org/pdf/2304.07193"""

    def __init__(self, rate, seed=None, **kwargs):
        kwargs.pop("noise_shape", None)
        super().__init__(rate=rate, seed=seed, **kwargs)
        self.input_spec = InputSpec(min_ndim=1)

    def call(self, inputs, training=False, **kwargs):
        batch_size = ops.shape(inputs)[0]
        indices = ops.arange(batch_size, dtype="int32")

        if 0.0 == self.rate or not training:
            return inputs, indices

        keep_size = ops.cast(batch_size, self.compute_dtype) * (1.0 - self.rate)
        keep_size = ops.ceil(keep_size / 8.0) * 8.0
        keep_size = ops.cast(keep_size, "int32")
        keep_size = ops.minimum(keep_size, batch_size)

        indices = ops.random.shuffle(indices, seed=self.seed_generator)

        outputs = ops.take(inputs, indices[:keep_size], axis=0)
        return outputs, indices

    def compute_output_shape(self, input_shape):
        if input_shape[0] is None:
            keep_size = None
        else:
            keep_size = input_shape[0] * (1.0 - self.rate)
            keep_size = int(math.ceil(keep_size / 8.0) * 8.0)
            keep_size = min(keep_size, input_shape[0])

        return (keep_size,) + input_shape[1:], input_shape[:1]

    def compute_output_spec(self, inputs, training=False):
        output_spec = super().compute_output_spec(inputs, training=training)

        return output_spec[0], KerasTensor(output_spec[1].shape, dtype="int32")

    def get_config(self):
        config = super().get_config()
        del config["noise_shape"]

        return config


@register_keras_serializable(package="SegMe>Common")
class RestorePath(layers.Layer):
    """Proposed in https://arxiv.org/pdf/2304.07193"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [
            InputSpec(min_ndim=1),
            InputSpec(ndim=1, dtype="int32"),
        ]

    def call(self, inputs, training=False, **kwargs):
        outputs, indices = inputs

        if not training:
            return outputs

        outputs_shape = ops.shape(outputs)
        batch_size = ops.size(indices)

        outputs = ops.cond(
            ops.equal(outputs_shape[0], batch_size),
            lambda: outputs,
            lambda: self.restore(outputs, indices, outputs_shape, batch_size),
        )

        return outputs

    def restore(self, outputs, indices, outputs_shape, batch_size):
        outputs *= ops.cast(batch_size, self.compute_dtype) / ops.cast(
            outputs_shape[0], self.compute_dtype
        )
        drops = ops.zeros(
            (batch_size - outputs_shape[0],) + outputs_shape[1:],
            dtype=outputs.dtype,
        )
        outputs = ops.concatenate([outputs, drops], axis=0)

        indices = ops.argsort(indices)
        outputs = ops.take(outputs, indices, axis=0)

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[1][:1] + input_shape[0][1:]


@register_keras_serializable(package="SegMe>Common")
class DropBlock(layers.Dropout):
    """Proposed in: https://arxiv.org/pdf/1810.12890"""

    def __init__(self, rate, size, seed=None, **kwargs):
        kwargs.pop("noise_shape", None)
        super().__init__(rate=rate, seed=seed, **kwargs)
        self.input_spec = InputSpec(ndim=4)

        if not 0.0 <= rate <= 1.0:
            raise ValueError(
                f"Invalid value {rate} received for `rate`. Expected a value "
                f"between 0 and 1."
            )
        if size < 1:
            raise ValueError(
                f"Invalid value {size} received for `size`. Expected a value "
                f"above 0."
            )

        self.rate = rate
        self.size = size

    def call(self, inputs, training=False, **kwargs):
        if 0.0 == self.rate or not training:
            return inputs

        shape = ops.shape(inputs)

        mask = backend.random.uniform(
            shape, dtype=self.compute_dtype, seed=self.seed_generator
        )
        mask = ops.cast(mask < self.rate / self.size**2, self.compute_dtype)
        mask = 1.0 - ops.max_pool(mask, self.size, strides=1, padding="same")
        mask = mask / ops.mean(mask, axis=[1, 2], keepdims=True)

        outputs = inputs * mask

        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({"size": self.size})
        del config["noise_shape"]

        return config
