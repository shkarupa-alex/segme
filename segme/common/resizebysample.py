import tensorflow as tf
from keras import layers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion


@register_keras_serializable(package='SegMe')
class ResizeBySample(layers.Layer):
    def __init__(self, method=tf.image.ResizeMethod.BILINEAR, antialias=False, **kwargs):
        kwargs['autocast'] = False
        super().__init__(**kwargs)
        self.input_spec = [
            layers.InputSpec(ndim=4),  # targets
            layers.InputSpec(ndim=4)  # samples
        ]

        self.method = method
        self.antialias = antialias

    def call(self, inputs, **kwargs):
        targets, samples = inputs

        new_size = tf.shape(samples)[1:3]
        resized = tf.image.resize(targets, new_size, method=self.method, antialias=self.antialias)

        targets_dtype = tf.dtypes.as_dtype(targets.dtype)
        if targets_dtype.is_integer:
            resized = tf.round(resized)
        resized = tf.cast(resized, targets.dtype)

        new_shape = targets.shape[0], samples.shape[1], samples.shape[2], targets.shape[3]
        resized.set_shape(new_shape)

        return resized

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        targets_shape, samples_shape = input_shape

        return (targets_shape[-0],) + samples_shape[1:3] + (targets_shape[3],)

    def compute_output_signature(self, input_signature):
        output_signature = super().compute_output_signature(input_signature)

        return tf.TensorSpec(dtype=input_signature[0].dtype, shape=output_signature.shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            'method': self.method,
            'antialias': self.antialias
        })

        return config


def resize_by_sample(inputs, **kwargs):
    return ResizeBySample(**kwargs)(inputs)
