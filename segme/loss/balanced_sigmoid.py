import tensorflow as tf
from tensorflow.python.keras.losses import LossFunctionWrapper


@tf.keras.utils.register_keras_serializable(package='SegMe')
class BalancedSigmoidCrossEntropy(LossFunctionWrapper):
    """ Proposed in: 'Holistically-Nested Edge Detection (CVPR 15)'

    Implements Equation [2] in https://arxiv.org/pdf/1504.06375.pdf
    Compute edge pixels for each training sample and set as pos_weights to tf.nn.weighted_cross_entropy_with_logits
    """
    def __init__(
            self, from_logits=False, reduction=tf.keras.losses.Reduction.AUTO, name='balanced_sigmoid_cross_entropy'):
        super().__init__(balanced_sigmoid_cross_entropy, reduction=reduction, name=name, from_logits=from_logits)


def balanced_sigmoid_cross_entropy(y_true, y_pred, from_logits):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, dtype=y_pred.dtype)

    ce = tf.keras.backend.binary_crossentropy(y_true, y_pred, from_logits=from_logits)

    total = tf.cast(tf.size(y_true), dtype=y_pred.dtype)
    negative = total - tf.reduce_sum(y_true)
    beta = negative / total
    beta_factor = y_true * beta + (1 - y_true) * (1. - beta)

    return tf.reduce_mean(beta_factor * ce, axis=-1)
