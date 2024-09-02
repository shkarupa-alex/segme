from keras.src import ops
from keras.src.saving import register_keras_serializable

from segme.loss.common_loss import crossentropy
from segme.loss.common_loss import to_1hot
from segme.loss.common_loss import to_probs
from segme.loss.common_loss import validate_input
from segme.loss.weighted_wrapper import WeightedLossFunctionWrapper


@register_keras_serializable(package="SegMe>Loss")
class CalibratedFocalCrossEntropy(WeightedLossFunctionWrapper):
    """Proposed in: 'Calibrating Deep Neural Networks using Focal Loss'

    Implements Equations from https://arxiv.org/pdf/2002.09437
    Note: remember to use focal loss trick: initialize last layer's bias with
    small negative value like -1.996
    """

    def __init__(
        self,
        prob0=0.2,
        prob1=0.5,
        gamma0=5.0,
        gamma1=3.0,
        from_logits=False,
        label_smoothing=0.0,
        force_binary=False,
        reduction="sum_over_batch_size",
        name="calibrated_focal_cross_entropy",
    ):
        super().__init__(
            calibrated_focal_cross_entropy,
            reduction=reduction,
            name=name,
            prob0=prob0,
            prob1=prob1,
            gamma0=gamma0,
            gamma1=gamma1,
            from_logits=from_logits,
            label_smoothing=label_smoothing,
            force_binary=force_binary,
        )


def calibrated_focal_cross_entropy(
    y_true,
    y_pred,
    sample_weight,
    prob0,
    prob1,
    gamma0,
    gamma1,
    from_logits,
    label_smoothing,
    force_binary,
):
    y_true, y_pred, sample_weight = validate_input(
        y_true, y_pred, sample_weight, dtype="int64", rank=4, channel="sparse"
    )
    y_prob, _ = to_probs(y_pred, from_logits, force_binary=False)
    y_true_1h, y_prob_1h = to_1hot(y_true, y_prob, False, dtype=y_prob.dtype)

    p_t = ops.sum(y_true_1h * y_prob_1h, axis=-1, keepdims=True)

    gamma = ops.where(p_t < prob0, gamma0, gamma1)
    gamma = ops.where(p_t >= prob1, 0.0, gamma)

    modulating_factor = ops.power(1.0 - p_t, gamma)

    sample_weight = (
        modulating_factor
        if sample_weight is None
        else modulating_factor * sample_weight
    )
    sample_weight = ops.stop_gradient(sample_weight)

    loss = crossentropy(
        y_true,
        y_pred,
        sample_weight,
        from_logits,
        label_smoothing=label_smoothing,
        force_binary=force_binary,
    )

    return loss
