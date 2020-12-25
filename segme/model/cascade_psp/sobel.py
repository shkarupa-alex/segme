import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, utils
from tensorflow.python.keras.utils.tf_utils import shape_type_conversion


@utils.register_keras_serializable(package='SegMe>CascadePSP')
class Sobel(layers.Layer):
    def __init__(self, epsilon, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)
        self.epsilon = epsilon

    @shape_type_conversion
    def build(self, input_shape):
        self.channels = input_shape[-1]
        if self.channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')
        self.input_spec = layers.InputSpec(ndim=4, axes={-1: self.channels})

        self.pool = layers.AveragePooling2D(3, strides=1, padding='same')

        x_kernel = np.reshape([[1, 0, -1], [2, 0, -2], [1, 0, -1]], [3, 3, 1, 1]) / 4
        x_kernel = np.tile(x_kernel, [1, 1, self.channels, 1])
        self.x_kernel = tf.constant(x_kernel, self.compute_dtype)

        y_kernel = np.reshape([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], [3, 3, 1, 1]) / 4
        y_kernel = np.tile(y_kernel, [1, 1, self.channels, 1])
        self.y_kernel = tf.constant(y_kernel, self.compute_dtype)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        pooled = self.pool(inputs)

        grad_x = tf.nn.depthwise_conv2d(pooled, self.x_kernel, strides=[1] * 4, padding='SAME')
        grad_y = tf.nn.depthwise_conv2d(pooled, self.y_kernel, strides=[1] * 4, padding='SAME')

        outputs = tf.sqrt(grad_x ** 2 + grad_y ** 2 + self.epsilon)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({'epsilon': self.epsilon})

        return config
