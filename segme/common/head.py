from tensorflow.keras import activations, layers, utils
from tensorflow.python.keras.utils.tf_utils import shape_type_conversion


class BaseHead2D(layers.Layer):
    def __init__(self, classes, activation, kernel_size=1, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)
        self.classes = classes
        self._classes = self.classes if self.classes > 2 else 1
        self.activation = activations.get(activation)
        self.kernel_size = kernel_size

    @shape_type_conversion
    def build(self, input_shape):
        self.pred = layers.Conv2D(self._classes, self.kernel_size, padding='same')
        self.act = layers.Activation(self.activation, dtype='float32')  # fp16

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = self.pred(inputs)
        outputs = self.act(outputs)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self._classes,)

    def get_config(self):
        config = super().get_config()
        config.update({
            'classes': self.classes,
            'activation': activations.serialize(self.activation),
            'kernel_size': self.kernel_size
        })

        return config


@utils.register_keras_serializable(package='SegMe')
class ClassificationHead2D(BaseHead2D):
    def __init__(self, classes, **kwargs):
        _activation = 'softmax' if classes > 2 else 'sigmoid'
        super().__init__(classes, _activation, **kwargs)

    def get_config(self):
        config = super().get_config()
        del config['activation']

        return config


@utils.register_keras_serializable(package='SegMe')
class RegressionHead2D(BaseHead2D):
    def __init__(self, **kwargs):
        super().__init__(1, 'linear', **kwargs)

    def get_config(self):
        config = super().get_config()
        del config['classes']
        del config['activation']

        return config