import tensorflow as tf
from keras import initializers, layers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion


@register_keras_serializable(package='SegMe>Common')
class HeadProjection(layers.Layer):
    def __init__(self, classes, kernel_size=1, kernel_initializer='glorot_uniform', **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)
        self.classes = classes
        self.kernel_size = kernel_size
        self.kernel_initializer = initializers.get(kernel_initializer)

    @shape_type_conversion
    def build(self, input_shape):
        self.proj = layers.Conv2D(
            self.classes, self.kernel_size, padding='same', kernel_initializer=self.kernel_initializer)

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


@register_keras_serializable(package='SegMe>Common')
class ClassificationActivation(layers.Layer):
    def __init__(self, **kwargs):
        kwargs['dtype'] = 'float32'
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(min_ndim=2)

    @shape_type_conversion
    def build(self, input_shape):
        channels = input_shape[-1]
        if channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')

        activation = 'softmax' if channels > 1 else 'sigmoid'
        self.act = layers.Activation(activation, dtype=self.dtype)  # for mixed_float16

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        return self.act(inputs)

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape


@register_keras_serializable(package='SegMe>Common')
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
            self.classes, kernel_size=self.kernel_size, kernel_initializer=self.kernel_initializer)
        self.act = ClassificationActivation()

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
        output_signature = self.act.compute_output_signature(output_signature)

        return output_signature

    def get_config(self):
        config = super().get_config()
        config.update({
            'classes': self.classes,
            'kernel_size': self.kernel_size,
            'kernel_initializer': initializers.serialize(self.kernel_initializer)
        })

        return config
