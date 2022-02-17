import tensorflow as tf
from keras import backend, losses
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.losses_utils import ReductionV2 as Reduction
from .common_loss import validate_input, crossentropy, to_probs, to_1hot


@register_keras_serializable(package='SegMe')
class NormalizedFocalCrossEntropy(losses.LossFunctionWrapper):
    """ Proposed in: 'AdaptIS: Adaptive Instance Selection Network'

    Implements Equations (Appendix A) from https://arxiv.org/pdf/1909.07829v1.pdf
    Note: remember to use focal loss trick: initialize last layer's bias with small negative value like -1.996
    """

    def __init__(
            self, from_logits=False, gamma=2, reduction=Reduction.AUTO,
            name='normalized_focal_cross_entropy'):
        super().__init__(
            normalized_focal_cross_entropy, reduction=reduction, name=name, from_logits=from_logits,
            gamma=gamma)


def normalized_focal_cross_entropy(y_true, y_pred, gamma, from_logits):
    y_true, y_pred, sample_weight = validate_input(
        y_true, y_pred, weight=None, dtype='int32', rank=None, channel='sparse')
    y_prob = to_probs(y_pred, from_logits, force_sigmoid=False)
    y_true_1h, y_prob_1h = to_1hot(y_true, y_prob)
    y_true_1h = tf.cast(y_true_1h, y_prob_1h.dtype)

    pt = tf.reduce_max(y_true_1h * y_prob_1h, axis=-1, keepdims=True)
    beta = (1. - pt) ** gamma
    axis_hw = list(range(1, y_true_1h.shape.ndims - 1))
    beta /= (tf.reduce_mean(beta, axis=axis_hw, keepdims=True) + backend.epsilon())
    beta = tf.stop_gradient(beta)

    loss = crossentropy(y_true, y_pred, beta, from_logits)

    return tf.reduce_mean(loss, axis=-1)
