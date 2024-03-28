import tensorflow as tf
from tf_keras.saving import register_keras_serializable
from tf_keras.src.utils.losses_utils import ReductionV2 as Reduction
from segme.loss.common_loss import validate_input, crossentropy, to_probs, to_1hot
from segme.loss.weighted_wrapper import WeightedLossFunctionWrapper


@register_keras_serializable(package='SegMe>Loss')
class CalibratedFocalCrossEntropy(WeightedLossFunctionWrapper):
    """ Proposed in: 'Calibrating Deep Neural Networks using Focal Loss'

    Implements Equations from https://arxiv.org/pdf/2002.09437.pdf
    Note: remember to use focal loss trick: initialize last layer's bias with small negative value like -1.996
    """

    def __init__(self, prob0=0.2, prob1=0.5, gamma0=5.0, gamma1=3.0, from_logits=False,
                 reduction=Reduction.AUTO, name='calibrated_focal_cross_entropy'):
        super().__init__(calibrated_focal_cross_entropy, reduction=reduction, name=name, prob0=prob0, prob1=prob1,
                         gamma0=gamma0, gamma1=gamma1, from_logits=from_logits)


def calibrated_focal_cross_entropy(y_true, y_pred, sample_weight, prob0, prob1, gamma0, gamma1, from_logits):
    y_true, y_pred, sample_weight = validate_input(
        y_true, y_pred, sample_weight, dtype='int64', rank=4, channel='sparse')
    y_prob = to_probs(y_pred, from_logits, force_sigmoid=False)
    y_true_1h, y_prob_1h = to_1hot(y_true, y_prob, dtype=y_prob.dtype)

    p_t = tf.reduce_sum(y_true_1h * y_prob_1h, axis=-1, keepdims=True)

    gamma = tf.where(tf.less(p_t, prob0), gamma0, gamma1)
    gamma = tf.where(tf.greater_equal(p_t, prob1), 0., gamma)

    modulating_factor = tf.pow(1. - p_t, gamma)

    sample_weight = modulating_factor if sample_weight is None else modulating_factor * sample_weight
    sample_weight = tf.stop_gradient(sample_weight)

    loss = crossentropy(y_true, y_pred, sample_weight, from_logits, False, 0.)

    return loss
