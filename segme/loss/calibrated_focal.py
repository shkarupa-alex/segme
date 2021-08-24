import tensorflow as tf
from keras import backend, losses
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.losses_utils import ReductionV2 as Reduction


@register_keras_serializable(package='SegMe')
class CalibratedFocalSigmoidCrossEntropy(losses.LossFunctionWrapper):
    """ Proposed in: 'Calibrating Deep Neural Networks using Focal Loss'

    Implements Equations from https://arxiv.org/pdf/2002.09437.pdf
    Note: remember to use focal loss trick: initialize last layer's bias with small negative value like -1.996
    """
    def __init__(
            self, from_logits=False, prob0=0.2, prob1=0.5, gamma0=5.0, gamma1=3.0, alpha=0.25,
            reduction=Reduction.AUTO, name='calibrated_focal_sigmoid_cross_entropy'):
        super().__init__(
            calibrated_focal_sigmoid_cross_entropy, reduction=reduction, name=name, from_logits=from_logits,
            prob0=prob0, prob1=prob1, gamma0=gamma0, gamma1=gamma1, alpha=alpha)


def calibrated_focal_sigmoid_cross_entropy(y_true, y_pred, prob0, prob1, gamma0, gamma1, alpha, from_logits):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, dtype=y_pred.dtype)

    ce = backend.binary_crossentropy(y_true, y_pred, from_logits=from_logits)
    pred_prob = y_pred if not from_logits else tf.sigmoid(y_pred)

    p_t = (y_true * pred_prob) + ((1 - y_true) * (1. - pred_prob))

    alpha = tf.convert_to_tensor(alpha, dtype=y_pred.dtype)
    alpha_factor = y_true * alpha + (1 - y_true) * (1. - alpha)

    prob0 = tf.convert_to_tensor(prob0, dtype=backend.floatx())
    prob1 = tf.convert_to_tensor(prob1, dtype=backend.floatx())
    gamma0 = tf.convert_to_tensor(gamma0, dtype=backend.floatx())
    gamma1 = tf.convert_to_tensor(gamma1, dtype=backend.floatx())
    gamma3 = tf.convert_to_tensor(0.0, dtype=backend.floatx())
    gamma = tf.where(tf.less(pred_prob, prob0), gamma0, gamma1)
    gamma = tf.where(tf.greater_equal(pred_prob, prob1), gamma3, gamma)

    modulating_factor = tf.pow(1.0 - p_t, gamma)

    return tf.reduce_mean(alpha_factor * modulating_factor * ce, axis=-1)
