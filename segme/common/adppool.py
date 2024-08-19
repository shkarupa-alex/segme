import numpy as np
import tensorflow as tf
from keras.src import layers
from keras.src.layers.input_spec import InputSpec
from keras.src.saving import register_keras_serializable
from keras.src.utils.argument_validation import standardize_tuple

from segme.common.shape import get_shape

# from keras.src.utils.control_flow_util import ops.cond


@register_keras_serializable(package="SegMe>Common")
class AdaptiveAveragePooling(layers.Layer):
    def __init__(self, output_size, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=4)
        self.output_size = standardize_tuple(output_size, 2, "output_size")

    def call(self, inputs, *args, **kwargs):
        if (1, 1) == self.output_size:
            return self.case_global(inputs)

        (height, width), static_size = get_shape(inputs, axis=[1, 2])
        if static_size:
            return self.case_static(inputs, height, width)

        return self.case_dynamic_nondivisible(inputs, height, width)

    def case_global(self, inputs):
        return tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)

    def case_static(self, inputs, height, width):
        start_h = (
            np.arange(self.output_size[0], dtype="float32")
            * height
            / self.output_size[0]
        )
        start_h = start_h.astype("int32")
        stop_h = (
            (np.arange(self.output_size[0], dtype="float32") + 1)
            * height
            / self.output_size[0]
        )
        stop_h = np.ceil(stop_h).astype("int32")
        size_h = stop_h - start_h
        over_h = stop_h[:-1] - start_h[1:]

        start_w = (
            np.arange(self.output_size[1], dtype="float32")
            * width
            / self.output_size[1]
        )
        start_w = start_w.astype("int32")
        stop_w = (
            (np.arange(self.output_size[1], dtype=np.float32) + 1)
            * width
            / self.output_size[1]
        )
        stop_w = np.ceil(stop_w).astype("int32")
        size_w = stop_w - start_w
        over_w = stop_w[:-1] - start_w[1:]

        kernels = np.array([size_h.max(), size_w.max()])
        if (kernels < 1).any():
            return self.case_static_nondivisible(inputs, height, width)
        if np.unique(size_h[1:-1]).size > 1 or np.unique(size_w[1:-1]).size > 1:
            return self.case_static_nondivisible(inputs, height, width)

        if np.unique(over_h).size > 1 or np.unique(over_w).size > 1:
            return self.case_static_nondivisible(inputs, height, width)
        strides = kernels - np.array([over_h.max(), over_w.max()])

        paddings = kernels - np.array([size_h.min(), size_w.min()])
        paddings_ = [
            [0, 0],
            [paddings[0], paddings[0]],
            [paddings[1], paddings[1]],
            [0, 0],
        ]

        outputs = tf.pad(inputs, paddings_)
        outputs = tf.nn.avg_pool(outputs, kernels, strides, "VALID")

        weights = tf.ones([1, height, width, 1], dtype=inputs.dtype)
        weights = tf.pad(weights, paddings_)
        weights = tf.nn.avg_pool(weights, kernels, strides, "VALID")
        outputs /= weights

        return outputs

    def case_static_nondivisible(self, inputs, height, width):
        start_h = (
            np.arange(self.output_size[0], dtype="float32")
            * height
            / self.output_size[0]
        )
        start_h = start_h.astype("int32")
        stop_h = (
            (np.arange(self.output_size[0], dtype="float32") + 1)
            * height
            / self.output_size[0]
        )
        stop_h = np.ceil(stop_h).astype("int32")

        pooled_h = []
        for idx in range(self.output_size[0]):
            pooled_h.append(
                tf.reduce_mean(
                    inputs[:, start_h[idx] : stop_h[idx]], axis=1, keepdims=True
                )
            )
        pooled_h = tf.concat(pooled_h, axis=1)

        start_w = (
            np.arange(self.output_size[1], dtype="float32")
            * width
            / self.output_size[1]
        )
        start_w = start_w.astype("int32")
        stop_w = (
            (np.arange(self.output_size[1], dtype=np.float32) + 1)
            * width
            / self.output_size[1]
        )
        stop_w = np.ceil(stop_w).astype("int32")

        pooled_w = []
        for idx in range(self.output_size[1]):
            pooled_w.append(
                tf.reduce_mean(
                    pooled_h[:, :, start_w[idx] : stop_w[idx]],
                    axis=2,
                    keepdims=True,
                )
            )
        pooled_w = tf.concat(pooled_w, axis=2)

        return pooled_w

    def case_dynamic_nondivisible(self, inputs, height, width):
        start_h = (
            tf.range(self.output_size[0], dtype="float32")
            * tf.cast(height, "float32")
            / self.output_size[0]
        )
        start_h = tf.cast(start_h, "int32")
        stop_h = (
            (tf.range(self.output_size[0], dtype="float32") + 1)
            * tf.cast(height, "float32")
            / self.output_size[0]
        )
        stop_h = tf.cast(tf.math.ceil(stop_h), "int32")

        pooled_h = []
        for idx in range(self.output_size[0]):
            pooled_h.append(
                tf.reduce_mean(
                    inputs[:, start_h[idx] : stop_h[idx]],
                    axis=1,
                    keepdims=True,
                )
            )
        pooled_h = tf.concat(pooled_h, axis=1)

        start_w = (
            tf.range(self.output_size[1], dtype="float32")
            * tf.cast(width, "float32")
            / self.output_size[1]
        )
        start_w = tf.cast(start_w, "int32")
        stop_w = (
            (tf.range(self.output_size[1], dtype="float32") + 1)
            * tf.cast(width, "float32")
            / self.output_size[1]
        )
        stop_w = tf.cast(tf.math.ceil(stop_w), "int32")

        pooled_w = []
        for idx in range(self.output_size[1]):
            pooled_w.append(
                tf.reduce_mean(
                    pooled_h[:, :, start_w[idx] : stop_w[idx]],
                    axis=2,
                    keepdims=True,
                )
            )
        pooled_w = tf.concat(pooled_w, axis=2)

        return pooled_w

    def compute_output_shape(self, input_shape):
        return (
            input_shape[0],
            self.output_size[0],
            self.output_size[1],
            input_shape[3],
        )

    def get_config(self):
        config = super().get_config()
        config.update({"output_size": self.output_size})

        return config
