import tensorflow as tf
from keras.src import backend
from keras.src import layers, ops
from keras.src.layers.input_spec import InputSpec
from keras.src.saving import register_keras_serializable

from segme.common.shape import get_shape


@register_keras_serializable(package="SegMe>Common")
class DropPath(layers.Dropout):
    def __init__(self, rate, seed=None, **kwargs):
        kwargs.pop("noise_shape", None)
        super().__init__(rate=rate, seed=seed, **kwargs)

    def call(self, inputs, training=False):
        if 0.0 == self.rate or not training:
            return inputs

        batch_size, _ = get_shape(inputs, axis=[0])
        noise_shape = batch_size + [1] * (inputs.shape.rank - 1)

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

# TODO
# @register_keras_serializable(package="SegMe>Common")
# class SlicePath(layers.Layer):
#     def __init__(self, rate, seed=None, **kwargs):
#         super().__init__(**kwargs)
#         self.input_spec = InputSpec(min_ndim=1)
#
#         if not 0.0 <= rate <= 1.0:
#             raise ValueError(
#                 f"Invalid value {rate} received for `rate`. Expected a value between 0 and 1."
#             )
#
#         self.rate = rate
#         self.seed = seed
#
#     def call(self, inputs, training=False, **kwargs):
#         [batch_size], _ = get_shape(inputs, axis=[0])
#
#         if 0.0 == self.rate:
#             return self.skip(inputs, batch_size)
#
#         if training:
#             return self.slice(inputs, batch_size)
#         else:
#             return self.skip(inputs, batch_size)
#
#     def skip(self, inputs, batch_size):
#         return tf.identity(inputs), tf.range(batch_size)
#
#     def slice(self, inputs, batch_size):
#         keep_size = tf.cast(batch_size, "float32") * (1.0 - self.rate)
#         keep_size = tf.math.ceil(keep_size / 8.0) * 8.0
#         keep_size = tf.cast(keep_size, "int32")
#         keep_size = tf.minimum(keep_size, batch_size)
#
#         indices = tf.range(batch_size)
#         indices = tf.random.shuffle(indices, self.seed)
#
#         outputs = tf.gather(inputs, indices[:keep_size], axis=0)
#
#         return outputs, indices
#
#     def compute_output_shape(self, input_shape):
#         return (None,) + input_shape[1:], input_shape[:1]
#
#     def get_config(self):
#         config = super().get_config()
#         config.update({"rate": self.rate, "seed": self.seed})
#
#         return config
#
#
# @register_keras_serializable(package="SegMe>Common")
# class RestorePath(layers.Layer):
#     def __init__(self, rate, seed=None, **kwargs):
#         super().__init__(seed=seed, **kwargs)
#         self.input_spec = [
#             InputSpec(min_ndim=1),
#             InputSpec(ndim=1, dtype="int32"),
#         ]
#
#         self.rate = rate
#         self.seed = seed
#
#     def call(self, inputs, training=False, **kwargs):
#         outputs, indices = inputs
#
#         if 0.0 == self.rate or not training:
#             return outputs
#
#         return self.restore(outputs, indices)
#
#     def restore(self, outputs, indices):
#         outputs_shape, _ = get_shape(outputs)
#         keep_size = outputs_shape[0]
#         batch_size = tf.size(indices)
#
#         keep_up = tf.cast(batch_size, "float32") / tf.cast(keep_size, "float32")
#         keep_min = (1.0 - self.rate) * keep_up
#         keep_max = (2.0 - self.rate) * keep_up
#         noise_shape = [keep_size] + [1] * (outputs.shape.rank - 1)
#         random_mask = self._random_generator.random_uniform(
#             noise_shape, minval=keep_min, maxval=keep_max
#         )
#
#         inv_keep = 1.0 / (1.0 - self.rate)
#         random_mask = tf.cast(random_mask >= 1.0, outputs.dtype) * inv_keep
#
#         drops = tf.zeros(
#             [batch_size - keep_size] + outputs_shape[1:], dtype=outputs.dtype
#         )
#         outputs = tf.concat([outputs * random_mask, drops], axis=0)
#
#         indices = tf.argsort(indices)
#         outputs = tf.gather(outputs, indices, axis=0)
#
#         return outputs
#
#     def compute_output_shape(self, input_shape):
#         return (None,) + input_shape[0][1:]
#
#     def get_config(self):
#         config = super().get_config()
#         config.update({"rate": self.rate, "seed": self.seed})
#
#         return config


@register_keras_serializable(package="SegMe>Common")
class DropBlock(layers.Dropout):
    """Proposed in: https://arxiv.org/pdf/1810.12890.pdf"""

    def __init__(self, rate, size, seed=None, **kwargs):
        kwargs.pop("noise_shape", None)
        super().__init__(rate=rate, seed=seed, **kwargs)
        self.input_spec = InputSpec(ndim=4)

        if not 0.0 <= rate <= 1.0:
            raise ValueError(
                f"Invalid value {rate} received for `rate`. Expected a value between 0 and 1."
            )
        if size < 1:
            raise ValueError(
                f"Invalid value {size} received for `size`. Expected a value above 0."
            )

        self.rate = rate
        self.size = size

    def call(self, inputs, training=False, **kwargs):
        if 0.0 == self.rate or not training:
            return inputs

        shape, _ = get_shape(inputs)
        mask = backend.random.uniform(shape, dtype=self.compute_dtype, seed=self.seed_generator)
        mask = tf.cast(mask < self.rate / self.size**2, self.compute_dtype)
        mask = 1.0 - tf.nn.max_pool2d(mask, self.size, 1, "SAME")
        mask = mask / tf.reduce_mean(mask, axis=[1, 2], keepdims=True)

        outputs = inputs * mask

        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({"rate": self.rate, "size": self.size})
        del config["noise_shape"]

        return config
