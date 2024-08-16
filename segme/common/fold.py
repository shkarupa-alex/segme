import tensorflow as tf
from keras.src import layers
from keras.src.layers.input_spec import InputSpec
from keras.src.saving import register_keras_serializable


@register_keras_serializable(package="SegMe>Common")
class Fold(layers.Layer):
    def __init__(self, size, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=4)

        self.size = int(size)

    def build(self, input_shape):
        channels = input_shape[-1]
        if channels is None:
            raise ValueError(
                "Channel dimension of the inputs should be defined. Found `None`."
            )

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = tf.nn.space_to_depth(inputs, self.size)

        return outputs

    def compute_output_shape(self, input_shape):
        output_shape = [
            input_shape[0],
            None if input_shape[1] is None else input_shape[1] // self.size,
            None if input_shape[2] is None else input_shape[2] // self.size,
            input_shape[3] * self.size**2,
        ]

        return output_shape

    def get_config(self):
        config = super().get_config()
        config.update({"size": self.size})

        return config


@register_keras_serializable(package="SegMe>Common")
class UnFold(layers.Layer):
    def __init__(self, size, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=4)

        self.size = int(size)

    def build(self, input_shape):
        channels = input_shape[-1]
        if channels is None:
            raise ValueError(
                "Channel dimension of the inputs should be defined. Found `None`."
            )
        if channels % self.size**2:
            raise ValueError(
                "Channel size must be divisible by unfold size^2 without reminder."
            )

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = tf.nn.depth_to_space(inputs, self.size)

        return outputs

    def compute_output_shape(self, input_shape):
        output_shape = [
            input_shape[0],
            None if input_shape[1] is None else input_shape[1] * self.size,
            None if input_shape[2] is None else input_shape[2] * self.size,
            input_shape[3] // self.size**2,
        ]

        return output_shape

    def get_config(self):
        config = super().get_config()
        config.update({"size": self.size})

        return config
