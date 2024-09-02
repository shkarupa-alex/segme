from keras.src import ops
from keras.src.saving import register_keras_serializable

from segme.loss.common_loss import validate_input
from segme.loss.common_loss import weighted_loss
from segme.loss.weighted_wrapper import WeightedLossFunctionWrapper


@register_keras_serializable(package="SegMe>Loss")
class SoftMeanAbsoluteError(WeightedLossFunctionWrapper):
    def __init__(
        self,
        beta=1.0,
        reduction="sum_over_batch_size",
        name="soft_mean_absolute_error",
    ):
        super().__init__(
            soft_mean_absolute_error, reduction=reduction, name=name, beta=beta
        )


def soft_mean_absolute_error(y_true, y_pred, sample_weight, beta):
    y_true, y_pred, sample_weight = validate_input(
        y_true, y_pred, sample_weight, dtype=None, rank=None, channel="same"
    )
    beta = ops.cast(beta, dtype=y_pred.dtype)

    error = y_pred - y_true
    abs_error = ops.abs(error)
    square_error = ops.square(error)

    loss = ops.where(
        abs_error < beta, square_error * (0.5 / beta), abs_error - 0.5 * beta
    )
    loss = weighted_loss(loss, sample_weight)

    return loss
