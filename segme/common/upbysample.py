import tensorflow as tf
from tensorflow.keras import layers, utils
from tensorflow.python.keras.utils.tf_utils import shape_type_conversion


@utils.register_keras_serializable(package='SegMe')
class UpBySample(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [
            layers.InputSpec(ndim=4),  # targets
            layers.InputSpec(ndim=4)  # samples
        ]

    def call(self, inputs, **kwargs):
        targets, samples = inputs

        size_before = tf.shape(samples)[1:3]
        upsampled = tf.compat.v1.image.resize(targets, size_before, method='bilinear', align_corners=True)

        return upsampled

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        targets_shape, samples_shape = input_shape

        return samples_shape[:-1] + (targets_shape[-1],)


def UpBySample_2d(inputs, **kwargs):
    return UpBySample(**kwargs)(inputs)
