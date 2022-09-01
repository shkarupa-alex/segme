import tensorflow as tf
from keras import losses
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.losses_utils import ReductionV2 as Reduction
from segme.loss.common_loss import validate_input, crossentropy, to_probs, to_1hot


@register_keras_serializable(package='SegMe>Loss')
class CalibratedFocalCrossEntropy(losses.LossFunctionWrapper):
    """ Proposed in: 'Calibrating Deep Neural Networks using Focal Loss'

    Implements Equations from https://arxiv.org/pdf/2002.09437.pdf
    Note: remember to use focal loss trick: initialize last layer's bias with small negative value like -1.996
    """

    def __init__(
            self, from_logits=False, prob0=0.2, prob1=0.5, gamma0=5.0, gamma1=3.0, alpha=0.25,
            reduction=Reduction.AUTO, name='calibrated_focal_cross_entropy'):
        super().__init__(
            calibrated_focal_cross_entropy, reduction=reduction, name=name, from_logits=from_logits,
            prob0=prob0, prob1=prob1, gamma0=gamma0, gamma1=gamma1, alpha=alpha)


def calibrated_focal_cross_entropy(y_true, y_pred, prob0, prob1, gamma0, gamma1, alpha, from_logits):
    y_true, y_pred, sample_weight = validate_input(
        y_true, y_pred, weight=None, dtype='int32', rank=None, channel='sparse')
    y_prob = to_probs(y_pred, from_logits, force_sigmoid=False)
    y_true_1h, y_prob_1h = to_1hot(y_true, y_prob)
    y_true_1h = tf.cast(y_true_1h, y_prob_1h.dtype)

    alpha_factor = tf.reduce_sum(y_true_1h * alpha, axis=-1, keepdims=True)

    gamma = tf.where(tf.less(y_prob_1h, prob0), gamma0, gamma1)
    gamma = tf.where(tf.greater_equal(y_prob_1h, prob1), 0., gamma)

    p_t = tf.reduce_sum(y_true_1h * y_prob_1h, axis=-1, keepdims=True)
    modulating_factor = tf.pow(1.0 - p_t, gamma)

    total_factor = tf.stop_gradient(alpha_factor * modulating_factor)

    loss = crossentropy(y_true, y_pred, sample_weight=total_factor, from_logits=from_logits)

    return tf.reduce_mean(loss, axis=-1)
