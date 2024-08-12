import tensorflow as tf
from keras.src import backend
from keras.src.saving import register_keras_serializable

from segme.loss.common_loss import crossentropy
from segme.loss.common_loss import to_1hot
from segme.loss.common_loss import to_probs
from segme.loss.common_loss import validate_input
from segme.loss.weighted_wrapper import WeightedLossFunctionWrapper


@register_keras_serializable(package="SegMe>Loss")
class NormalizedFocalCrossEntropy(WeightedLossFunctionWrapper):
    """Proposed in: 'AdaptIS: Adaptive Instance Selection Network'

    Implements Equations (Appendix A) from https://arxiv.org/pdf/1909.07829v1.pdf
    Note: remember to use focal loss trick: initialize last layer's bias with small negative value like -1.996
    """

    def __init__(
        self,
        gamma=2,
        from_logits=False,
        reduction="sum_over_batch_size",
        name="normalized_focal_cross_entropy",
    ):
        super().__init__(
            normalized_focal_cross_entropy,
            reduction=reduction,
            name=name,
            gamma=gamma,
            from_logits=from_logits,
        )


def normalized_focal_cross_entropy(
    y_true, y_pred, sample_weight, gamma, from_logits
):
    y_true, y_pred, sample_weight = validate_input(
        y_true, y_pred, sample_weight, dtype="int64", rank=4, channel="sparse"
    )
    y_prob, _ = to_probs(y_pred, from_logits, force_binary=False)
    y_true_1h, y_prob_1h = to_1hot(y_true, y_prob, False, dtype=y_prob.dtype)

    pt = tf.reduce_max(y_true_1h * y_prob_1h, axis=-1, keepdims=True)
    beta = (1.0 - pt) ** gamma
    beta /= tf.reduce_mean(beta, axis=[1, 2], keepdims=True) + backend.epsilon()
    beta = tf.stop_gradient(beta)
    if sample_weight is not None:
        beta *= sample_weight

    loss = crossentropy(y_true, y_pred, beta, from_logits, 0.0, False)

    return loss
