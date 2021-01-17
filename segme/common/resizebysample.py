import tensorflow as tf
from tensorflow.keras import layers, utils
from tensorflow.python.keras.utils.tf_utils import shape_type_conversion


@utils.register_keras_serializable(package='SegMe')
class ResizeBySample(layers.Layer):
    def __init__(self, method='bilinear', align_corners=True, **kwargs):
        kwargs['autocast'] = False
        super().__init__(**kwargs)
        self.input_spec = [
            layers.InputSpec(ndim=4),  # targets
            layers.InputSpec(ndim=4)  # samples
        ]

        self.method = method
        self.align_corners = align_corners

    def call(self, inputs, **kwargs):
        targets, samples = inputs

        new_size = tf.shape(samples)[1:3]
        resized = tf.compat.v1.image.resize(targets, new_size, method=self.method, align_corners=self.align_corners)

        targets_dtype = tf.dtypes.as_dtype(targets.dtype)
        if targets_dtype.is_integer:
            result = tf.round(result)
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
            'align_corners': self.align_corners
        })

        return config


def resize_by_sample(inputs, **kwargs):
    return ResizeBySample(**kwargs)(inputs)
