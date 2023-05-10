import tensorflow as tf
from keras import layers
from keras.saving import register_keras_serializable
from keras.src.utils.tf_utils import shape_type_conversion
from segme.common.resize import NearestInterpolation


@register_keras_serializable(package='SegMe>Common>Align>FADE')
class CarafeConvolution(layers.Layer):
    def __init__(self, kernel_size, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [
            layers.InputSpec(ndim=4),  # features
            layers.InputSpec(ndim=4)]  # mask

        self.kernel_size = kernel_size

    @shape_type_conversion
    def build(self, input_shape):
        self.channels = [shape[-1] for shape in input_shape]
        if None in self.channels:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')
        self.input_spec = [
            layers.InputSpec(ndim=4, axes={-1: self.channels[0]}),
            layers.InputSpec(ndim=4, axes={-1: self.channels[1]})]

        self.internear = NearestInterpolation()

        self.group_size = self.channels[1] // (self.kernel_size ** 2)
        if self.group_size < 1 or self.channels[1] != self.group_size * self.kernel_size ** 2:
            raise ValueError('Wrong mask channel dimension.')

        if self.channels[0] % self.group_size:
            raise ValueError('Unable to split features into groups.')

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        features, masks = inputs

        batch, height, width, _ = tf.unstack(tf.shape(masks))
        output_shape = self.compute_output_shape([features.shape, masks.shape])

        features = tf.image.extract_patches(
            features, [1, self.kernel_size, self.kernel_size, 1], [1] * 4, [1] * 4, 'SAME')
        features = self.internear([features, masks])

        features = tf.reshape(
            features,
            (batch, height, width, self.group_size, self.channels[0] // self.group_size, self.kernel_size ** 2))
        masks = tf.reshape(masks, (batch, height, width, self.group_size, 1, self.kernel_size ** 2))

        outputs = tf.matmul(features, masks, transpose_b=True)

        outputs = tf.reshape(outputs, (batch, height, width, self.channels[0]))
        outputs.set_shape(output_shape)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[1][:-1] + (self.channels[0],)

    def get_config(self):
        config = super().get_config()
        config.update({'kernel_size': self.kernel_size})

        return config
