from keras.src.saving import register_keras_serializable

from segme.loss.common_loss import crossentropy
from segme.loss.common_loss import validate_input
from segme.loss.weighted_wrapper import WeightedLossFunctionWrapper


@register_keras_serializable(package="SegMe>Loss")
class CrossEntropyLoss(WeightedLossFunctionWrapper):
    def __init__(
        self,
        from_logits=False,
        label_smoothing=0.0,
        force_binary=False,
        reduction="sum_over_batch_size",
        name="cross_entropy_loss",
    ):
        super().__init__(
            cross_entropy_loss,
            reduction=reduction,
            name=name,
            from_logits=from_logits,
            force_binary=force_binary,
            label_smoothing=label_smoothing,
        )


def cross_entropy_loss(
    y_true, y_pred, sample_weight, from_logits, label_smoothing, force_binary
):
    y_true, y_pred, sample_weight = validate_input(
        y_true, y_pred, sample_weight, dtype=None, rank=None, channel=None
    )

    return crossentropy(
        y_true,
        y_pred,
        sample_weight,
        from_logits,
        label_smoothing,
        force_binary,
    )
