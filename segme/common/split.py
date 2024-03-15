import tensorflow as tf
from tf_keras import layers
from tf_keras.saving import register_keras_serializable
from tf_keras.src.utils.tf_utils import shape_type_conversion


@register_keras_serializable(package='SegMe>Common')
class Split(layers.Layer):
    def __init__(self, num_or_size_splits, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

        if not isinstance(num_or_size_splits, (int, list, tuple)):
            raise ValueError(
                f'Expected type of `num_or_size_splits` to be `int`, `list` or `tuple`. '
                f'Got {type(self.num_or_size_splits)}')

        self.num_or_size_splits = num_or_size_splits
        self.axis = axis

    @shape_type_conversion
    def build(self, input_shape):
        channels = input_shape[self.axis]
        if channels is None:
            raise ValueError('Split dimension of the inputs should be defined. Found `None`.')

        if isinstance(self.num_or_size_splits, int) and channels % self.num_or_size_splits:
            raise ValueError('Channel dimension of the inputs should be divi. Found `None`.')
        if isinstance(self.num_or_size_splits, (list, tuple)) and sum(self.num_or_size_splits) != channels:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')

        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        return tf.split(inputs, self.num_or_size_splits, self.axis)

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)

        if isinstance(self.num_or_size_splits, int):
            output_shape[self.axis] = input_shape[self.axis] // self.num_or_size_splits

            return [tuple(output_shape)] * self.num_or_size_splits

        output_shapes = []
        for size in self.num_or_size_splits:
            output_shape[self.axis] = size
            output_shapes.append(tuple(output_shape))

        return output_shapes

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_or_size_splits': self.num_or_size_splits,
            'axis': self.axis
        })

        return config
