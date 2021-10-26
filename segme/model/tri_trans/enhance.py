import tensorflow as tf
from keras import layers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from ...common import AdaptiveMaxPooling


@register_keras_serializable(package='SegMe>TriTrans')
class CAEnhance(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [
            layers.InputSpec(ndim=4),  # rgb
            layers.InputSpec(ndim=4)  # depth
        ]

    @shape_type_conversion
    def build(self, input_shape):
        channels0 = input_shape[0][-1]
        channels1 = input_shape[1][-1]
        if channels0 is None or channels1 is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')
        if channels1 < 8:
            raise ValueError('Channel dimension of depth input should be greater than 7.')

        self.input_spec = [
            layers.InputSpec(ndim=4, axes={-1: channels0}),  # rgb
            layers.InputSpec(ndim=4, axes={-1: channels1})  # depth
        ]

        self.pool = AdaptiveMaxPooling(1)

        self.conv1 = layers.Conv2D(channels1 // 8, 1, activation='relu', use_bias=False)
        self.conv2 = layers.Conv2D(channels1, 1, activation='sigmoid', use_bias=False)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        rgb, depth = inputs

        outputs = layers.concatenate([rgb, depth], axis=-1)
        outputs = self.pool(outputs)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        outputs *= depth

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[1]


@register_keras_serializable(package='SegMe>TriTrans')
class SAEnhance(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

    @shape_type_conversion
    def build(self, input_shape):
        self.conv = layers.Conv2D(1, 7, padding='same', activation='sigmoid', use_bias=False)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = tf.math.reduce_max(inputs, axis=-1, keepdims=True)
        outputs = self.conv(outputs)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (1,)


@register_keras_serializable(package='SegMe>TriTrans')
class CASAEnhance(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [
            layers.InputSpec(ndim=4),  # rgb
            layers.InputSpec(ndim=4)  # depth
        ]

    @shape_type_conversion
    def build(self, input_shape):
        self.ca = CAEnhance()
        self.sa = SAEnhance()

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        rgb, depth = inputs
        outputs = self.ca([rgb, depth])
        outputs = self.sa(outputs)
        outputs *= depth

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[1]
