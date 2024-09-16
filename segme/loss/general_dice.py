from keras.src import backend
from keras.src import ops
from keras.src.saving import register_keras_serializable

from segme.loss.common_loss import iou
from segme.loss.common_loss import validate_input
from segme.loss.weighted_wrapper import WeightedLossFunctionWrapper


@register_keras_serializable(package="SegMe>Loss")
class GeneralizedDiceLoss(WeightedLossFunctionWrapper):
    """Proposed in: 'Generalised Dice overlap as a deep learning loss function
    for highly unbalanced segmentations'

    Implements Equations from https://arxiv.org/pdf/1707.03237v3.pdf
    """

    def __init__(
        self,
        from_logits=False,
        reduction="sum_over_batch_size",
        name="generalized_dice_loss",
    ):
        super().__init__(
            generalized_dice_loss,
            reduction=reduction,
            name=name,
            from_logits=from_logits,
        )


def generalized_dice_loss(y_true, y_pred, sample_weight, from_logits):
    y_true, y_pred, sample_weight = validate_input(
        y_true, y_pred, sample_weight, dtype="int64", rank=4, channel="sparse"
    )

    y_true_1h = ops.one_hot(
        ops.squeeze(y_true, -1), max(2, y_pred.shape[-1]), dtype=y_pred.dtype
    )
    weight = ops.square(ops.mean(y_true_1h, axis=[0, 1, 2], keepdims=True))
    weight = (
        ops.max(weight * y_true_1h, axis=-1, keepdims=True) + backend.epsilon()
    )
    weight = 1.0 / weight

    sample_weight = weight if sample_weight is None else sample_weight * weight
    sample_weight = ops.stop_gradient(sample_weight)

    loss = iou(
        y_true,
        y_pred,
        sample_weight,
        from_logits=from_logits,
        smooth=1.0,
        dice=True,
    )

    return loss
