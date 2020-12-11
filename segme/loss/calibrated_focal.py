import tensorflow as tf
from tensorflow.python.keras.losses import LossFunctionWrapper


@tf.keras.utils.register_keras_serializable(package='SegMe')
class SigmoidCalibratedFocalCrossEntropy(LossFunctionWrapper):
    """ Reference: https://arxiv.org/pdf/2002.09437.pdf

    Note: remember to use focal loss trick: initialize last layer's bias with small negative value like -1.996
    """
    def __init__(
            self, from_logits=False, prob0=0.2, gamma0=5.0, gamma1=3.0, alpha=0.25,
            reduction=tf.keras.losses.Reduction.AUTO, name='sigmoid_calibrated_focal_crossentropy'):
        super().__init__(
            sigmoid_calibrated_focal_crossentropy, reduction=reduction, name=name, from_logits=from_logits,
            prob0=prob0, gamma0=gamma0, gamma1=gamma1, alpha=alpha)


@tf.keras.utils.register_keras_serializable(package='SegMe')
def sigmoid_calibrated_focal_crossentropy(
        y_true, y_pred, prob0=0.2, gamma0=5.0, gamma1=3.0, alpha=0.25, from_logits=False):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, dtype=y_pred.dtype)

    ce = tf.keras.backend.binary_crossentropy(y_true, y_pred, from_logits=from_logits)
    pred_prob = y_pred if not from_logits else tf.sigmoid(y_pred)

    p_t = (y_true * pred_prob) + ((1 - y_true) * (1. - pred_prob))

    alpha = tf.convert_to_tensor(alpha, dtype=y_pred.dtype)
    alpha_factor = y_true * alpha + (1 - y_true) * (1. - alpha)

    prob0 = tf.convert_to_tensor(prob0, dtype=tf.keras.backend.floatx())
    gamma0 = tf.convert_to_tensor(gamma0, dtype=tf.keras.backend.floatx())
    gamma1 = tf.convert_to_tensor(gamma1, dtype=tf.keras.backend.floatx())
    gamma = tf.where(tf.less(y_pred, prob0), gamma0, gamma1)
    modulating_factor = tf.pow((1.0 - p_t), gamma)

    return tf.reduce_mean(alpha_factor * modulating_factor * ce, axis=-1)
