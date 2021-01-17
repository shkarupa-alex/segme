import tensorflow as tf
from tensorflow.keras import layers, utils
from tensorflow.python.keras.utils.tf_utils import shape_type_conversion


@utils.register_keras_serializable(package='SegMe')
class ResizeByScale(layers.Layer):
    def __init__(self, scale, method='bilinear', align_corners=True, **kwargs):
        kwargs['autocast'] = False
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)
        self.scale = float(scale)
        self.method = method
        self.align_corners = align_corners

    def _scale(self, value):
        return None if value is None else int(round(value * self.scale))

    def call(self, inputs, **kwargs):
        if 1 == self.scale:
            return inputs

        new_size = tf.cast(tf.round(tf.cast(tf.shape(inputs)[1:3], self.compute_dtype) * self.scale), 'int32')
        resized = tf.compat.v1.image.resize(inputs, new_size, method=self.method, align_corners=self.align_corners)

        inputs_dtype = tf.dtypes.as_dtype(inputs.dtype)
        if inputs_dtype.is_integer:
            resized = tf.round(resized)
        resized = tf.cast(resized, inputs.dtype)

        new_shape = inputs.shape[0], self._scale(inputs.shape[1]), self._scale(inputs.shape[2]), inputs.shape[3]
        resized.set_shape(new_shape)

        return resized

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        if 1 == self.scale:
            return input_shape

        return input_shape[0], self._scale(input_shape[1]), self._scale(input_shape[2]), input_shape[3]

    def compute_output_signature(self, input_signature):
        output_signature = super().compute_output_signature(input_signature)

        return tf.TensorSpec(dtype=input_signature.dtype, shape=output_signature.shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            'scale': self.scale,
            'method': self.method,
            'align_corners': self.align_corners
        })

        return config


def resize_by_scale(inputs, **kwargs):
    return ResizeByScale(**kwargs)(inputs)
