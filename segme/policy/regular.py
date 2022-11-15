import tensorflow as tf
from keras.regularizers import Regularizer, _check_penalty_number
from keras.utils.generic_utils import register_keras_serializable


@register_keras_serializable(package='SegMe>Policy>Regularizers')
class StandardizedRegularizer(Regularizer):
    def __init__(self, l1=1e-4):
        _check_penalty_number(l1)

        self.l1 = float(l1)

    def __call__(self, x):
        # Kernel has shape HWIO, normalize over HWI
        mean, var = tf.nn.moments(x, axes=[0, 1, 2], keepdims=True)
        y = tf.nn.batch_normalization(x, mean, var, None, None, variance_epsilon=1e-5)
        y = tf.stop_gradient(y)

        return self.l1 * tf.reduce_sum(tf.abs(x - y))

    def get_config(self):
        return {'l1': self.l1}
