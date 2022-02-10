import tensorflow as tf
from keras import backend, losses
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.losses_utils import ReductionV2 as Reduction


@register_keras_serializable(package='SegMe')
class BalancedSigmoidCrossEntropy(losses.LossFunctionWrapper):
    """ Proposed in: 'Holistically-Nested Edge Detection (CVPR 15)'

    Implements Equation [2] in https://arxiv.org/pdf/1504.06375.pdf
    """
    def __init__(
            self, from_logits=False, reduction=Reduction.AUTO, name='balanced_sigmoid_cross_entropy'):
        super().__init__(balanced_sigmoid_cross_entropy, reduction=reduction, name=name, from_logits=from_logits)


def balanced_sigmoid_cross_entropy(y_true, y_pred, from_logits):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, dtype=y_pred.dtype)

    ce = backend.binary_crossentropy(y_true, y_pred, from_logits=from_logits)

    total = tf.cast(tf.size(y_true), dtype=y_pred.dtype)
    negative = total - tf.reduce_sum(y_true)
    beta = negative / total
    beta_factor = y_true * beta + (1. - y_true) * (1. - beta)
    beta_factor = tf.stop_gradient(beta_factor)

    return tf.reduce_mean(beta_factor * ce, axis=-1)
