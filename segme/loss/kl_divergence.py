from keras.src import ops
from keras.src.saving import register_keras_serializable

from segme.loss.common_loss import validate_input
from segme.loss.common_loss import weighted_loss
from segme.loss.weighted_wrapper import WeightedLossFunctionWrapper


@register_keras_serializable(package="SegMe>Loss")
class KLDivergenceLoss(WeightedLossFunctionWrapper):
    def __init__(
        self,
        temperature=1.0,
        reduction="sum_over_batch_size",
        name="kl_divergence_loss",
    ):
        super().__init__(
            kl_divergence_loss,
            reduction=reduction,
            name=name,
            temperature=temperature,
        )


def kl_divergence_loss(y_true, y_pred, sample_weight, temperature):
    y_true, y_pred, sample_weight = validate_input(
        y_true, y_pred, sample_weight, dtype=None, rank=None, channel="same"
    )

    loss = _kl_divergence(y_true, y_pred, temperature)

    return weighted_loss(loss, sample_weight)


def _kl_divergence(y_true, y_pred, temperature):
    y_true *= 1.0 / temperature
    y_pred *= 1.0 / temperature

    y_true_sm = ops.softmax(y_true)
    y_true_sm = ops.stop_gradient(y_true_sm)
    y_true_lsm = ops.log_softmax(y_true)
    y_true_lsm = ops.stop_gradient(y_true_lsm)

    loss = y_true_sm * (y_true_lsm - ops.log_softmax(y_pred))
    loss = ops.sum(loss, axis=-1, keepdims=True)

    return loss
