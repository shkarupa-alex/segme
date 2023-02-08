import tensorflow as tf
from keras import layers
from keras.saving.object_registration import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion


@register_keras_serializable(package='SegMe>Common')
class UnFold(layers.Layer):
    def __init__(self, size, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        self.size = size

    @shape_type_conversion
    def build(self, input_shape):
        channels = input_shape[-1]
        if channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')
        if channels % self.size:
            raise ValueError('Channel size must be divisible by unfold size without reminder.')

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = tf.nn.depth_to_space(inputs, self.size)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        output_shape = [input_shape[0]]
        output_shape.append(None if input_shape[1] is None else input_shape[1] * self.size)
        output_shape.append(None if input_shape[2] is None else input_shape[2] * self.size)
        output_shape.append(input_shape[3] // self.size ** 2)

        return output_shape

    def get_config(self):
        config = super().get_config()
        config.update({'size': self.size})

        return config
