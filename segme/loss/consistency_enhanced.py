from keras.src import backend
from keras.src.saving import register_keras_serializable

from segme.loss.common_loss import to_1hot
from segme.loss.common_loss import to_probs
from segme.loss.common_loss import validate_input
from segme.loss.common_loss import weighted_loss
from segme.loss.weighted_wrapper import WeightedLossFunctionWrapper


@register_keras_serializable(package="SegMe>Loss")
class ConsistencyEnhancedLoss(WeightedLossFunctionWrapper):
    """Proposed in: 'Multi-scale Interactive Network for Salient Object
    Detection'

    Implements Equation [9] in https://arxiv.org/pdf/2007.09062.pdf
    """

    def __init__(
        self,
        from_logits=False,
        force_binary=False,
        reduction="sum_over_batch_size",
        name="consistency_enhanced_loss",
    ):
        super().__init__(
            consistency_enhanced_loss,
            reduction=reduction,
            name=name,
            from_logits=from_logits,
            force_binary=force_binary,
        )


def consistency_enhanced_loss(
    y_true, y_pred, sample_weight, from_logits, force_binary
):
    y_true, y_pred, sample_weight = validate_input(
        y_true, y_pred, sample_weight, dtype="int64", rank=4, channel="sparse"
    )
    y_pred, from_logits = to_probs(
        y_pred, from_logits, force_binary=force_binary
    )
    y_true, y_pred = to_1hot(y_true, y_pred, from_logits, dtype=y_pred.dtype)

    intersection = y_true * y_pred
    numerator0 = weighted_loss(y_pred - intersection, sample_weight)
    numerator1 = weighted_loss(y_true - intersection, sample_weight)
    denominator0 = weighted_loss(y_pred, sample_weight)
    denominator1 = weighted_loss(y_true, sample_weight)

    epsilon = backend.epsilon()
    loss = numerator0 + numerator1
    loss /= denominator0 + denominator1 + epsilon

    return loss
