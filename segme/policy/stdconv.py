import tensorflow as tf
from keras import backend, layers
from keras.utils.control_flow_util import smart_cond
from keras.utils.generic_utils import register_keras_serializable


@register_keras_serializable(package='SegMe>Policy')
class StandardizedConv2D(layers.Conv2D):
    """Implements https://arxiv.org/abs/1903.10520"""

    def normalize_call(self, inputs):
        kernel = tf.cast(self.kernel, self.dtype)

        # Kernel has shape HWIO, normalize over HWI
        mean, var = tf.nn.moments(kernel, axes=[0, 1, 2], keepdims=True)
        kernel = tf.nn.batch_normalization(kernel, mean, var, None, None, variance_epsilon=1e-5)
        kernel = tf.stop_gradient(kernel)

        with tf.control_dependencies([self.kernel.assign(kernel)]):
            outputs = tf.identity(inputs)

        return outputs

    def call(self, inputs, training=None):
        if training is None:
            training = backend.learning_phase()

        outputs = smart_cond(
            training,
            lambda: self.normalize_call(inputs),
            lambda: tf.identity(inputs))

        outputs = super().call(outputs)

        return outputs
