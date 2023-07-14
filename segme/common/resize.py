import numpy as np
import tensorflow as tf
from keras import layers
from keras.saving import register_keras_serializable
from keras.src.utils.tf_utils import shape_type_conversion
from segme.common.shape import get_shape


@register_keras_serializable(package='SegMe>Common>Interpolation')
class NearestInterpolation(layers.Layer):
    def __init__(self, scale=None, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4) if scale is not None else [
            layers.InputSpec(ndim=4), layers.InputSpec(ndim=4)]  # targets, samples

        self.scale = None if scale is None else float(scale)

    def resize(self, inputs, size):
        return tf.image.resize(inputs, size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    def call(self, inputs, **kwargs):
        if 1. == self.scale:
            return inputs

        if self.scale is None:
            targets, samples = inputs
            new_size, static_size = get_shape(samples, axis=[1, 2])
        else:
            targets = inputs
            new_size, static_size = get_shape(targets, axis=[1, 2])

            if static_size:
                new_size = np.array(new_size) * self.scale
                new_size = np.round(new_size).astype('int32')
            else:
                new_size = tf.cast(new_size, self.compute_dtype) * self.scale
                new_size = tf.cast(tf.round(new_size), 'int32')

        if (1, 1) == targets.shape[1:3]:
            repeats = [1] + new_size + [1] if static_size else tf.concat([[1], new_size, [1]], axis=-1)
            outputs = tf.tile(targets, repeats)
        else:
            outputs = self.resize(targets, new_size)

        if self.scale is None:
            outputs.set_shape(self.compute_output_shape([targets.shape, samples.shape]))
        else:
            outputs.set_shape(self.compute_output_shape(inputs.shape))

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        if 1 == self.scale:
            return input_shape

        if self.scale is None:
            targets_shape, samples_shape = input_shape
            return (targets_shape[-0],) + samples_shape[1:3] + (targets_shape[3],)

        def _scale(value):
            return None if value is None else int(round(value * self.scale))

        return input_shape[0], _scale(input_shape[1]), _scale(input_shape[2]), input_shape[3]

    def get_config(self):
        config = super().get_config()
        config.update({'scale': self.scale})

        return config


@register_keras_serializable(package='SegMe>Common>Interpolation')
class BilinearInterpolation(NearestInterpolation):
    def __init__(self, scale=None, compat=False, **kwargs):
        super().__init__(scale=scale, **kwargs)

        self.compat = compat

    def resize(self, inputs, size):
        outputs = tf.image.resize(inputs, size, method=tf.image.ResizeMethod.BILINEAR)
        outputs = tf.saturate_cast(outputs, self.compute_dtype)

        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({'compat': self.compat})

        return config
