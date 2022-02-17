import tensorflow as tf
from keras import initializers, layers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from .sameconv import SameConv


@register_keras_serializable(package='SegMe')
class HeadProjection(layers.Layer):
    def __init__(self, classes, kernel_size=1, kernel_initializer='glorot_uniform', **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)
        self.classes = classes
        self.kernel_size = kernel_size
        self.kernel_initializer = initializers.get(kernel_initializer)

    @shape_type_conversion
    def build(self, input_shape):
        self.proj = SameConv(self.classes, self.kernel_size, kernel_initializer=self.kernel_initializer)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        return self.proj(inputs)

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.classes,)

    def get_config(self):
        config = super().get_config()
        config.update({
            'classes': self.classes,
            'kernel_size': self.kernel_size,
            'kernel_initializer': initializers.serialize(self.kernel_initializer)
        })

        return config


@register_keras_serializable(package='SegMe')
class HeadActivation(layers.Layer):
    def __init__(self, classes, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)
        self.classes = classes

    @shape_type_conversion
    def build(self, input_shape):
        activation = 'softmax' if self.classes > 1 else 'sigmoid'
        self.act = layers.Activation(activation, dtype='float32')  # for mixed_float16

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        return self.act(inputs)

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_output_signature(self, input_signature):
        output_signature = super().compute_output_signature(input_signature)

        return tf.TensorSpec(dtype='float32', shape=output_signature.shape)

    def get_config(self):
        config = super().get_config()
        config.update({'classes': self.classes})

        return config


@register_keras_serializable(package='SegMe')
class ClassificationHead(layers.Layer):
    def __init__(self, classes, kernel_size=1, kernel_initializer='glorot_uniform', **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)
        self.classes = classes
        self.kernel_size = kernel_size
        self.kernel_initializer = initializers.get(kernel_initializer)

    @shape_type_conversion
    def build(self, input_shape):
        self.proj = HeadProjection(
            self.classes,
            kernel_size=self.kernel_size,
            kernel_initializer=self.kernel_initializer)
        self.act = HeadActivation(self.classes)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = self.proj(inputs)
        outputs = self.act(outputs)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.classes,)

    def compute_output_signature(self, input_signature):
        output_signature = super().compute_output_signature(input_signature)

        return tf.TensorSpec(dtype='float32', shape=output_signature.shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            'classes': self.classes,
            'kernel_size': self.kernel_size,
            'kernel_initializer': initializers.serialize(self.kernel_initializer)
        })

        return config
