from keras.src import layers
from keras.src.saving import register_keras_serializable


@register_keras_serializable(package="SegMe>Common")
class OneMinus(layers.Layer):
    def call(self, inputs, **kwargs):
        return 1.0 - inputs

    def compute_output_shape(self, input_shape):
        return input_shape
