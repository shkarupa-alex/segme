from keras import ops
from keras.src.saving import register_keras_serializable

from segme.loss.common_loss import crossentropy
from segme.loss.common_loss import iou
from segme.loss.common_loss import mae
from segme.loss.common_loss import validate_input
from segme.loss.weighted_wrapper import WeightedLossFunctionWrapper


@register_keras_serializable(package="SegMe>Loss")
class AdaptivePixelIntensityLoss(WeightedLossFunctionWrapper):
    """Proposed in:
        "TRACER: Extreme Attention Guided Salient Object Tracing Network"

    Implements Equation (12) from https://arxiv.org/pdf/2112.07380
    """

    def __init__(
        self,
        from_logits=False,
        label_smoothing=0.0,
        force_binary=False,
        reduction="sum_over_batch_size",
        name="adaptive_pixel_intensity_loss",
    ):
        super().__init__(
            adaptive_pixel_intensity_loss,
            reduction=reduction,
            name=name,
            from_logits=from_logits,
            label_smoothing=label_smoothing,
            force_binary=force_binary,
        )


def adaptive_pixel_intensity_loss(
    y_true, y_pred, sample_weight, from_logits, label_smoothing, force_binary
):
    y_true, y_pred, sample_weight = validate_input(
        y_true, y_pred, sample_weight, dtype="int64", rank=4, channel="sparse"
    )

    height, width = ops.shape(y_true)[1:3]
    if isinstance(height, int) and isinstance(width, int):
        if height <= 30 or width <= 30:
            raise ValueError(
                "Adaptive intensity loss does not support "
                "inputs with spatial size <= 30."
            )

    y_true_1h = ops.one_hot(
        ops.squeeze(y_true, -1),
        max(2, y_pred.shape[-1]),
        dtype=y_pred.dtype,
    )
    omega = (
        sum(
            [
                ops.abs(
                    ops.average_pool(y_true_1h, k, strides=1, padding="same")
                    - y_true_1h
                )
                for k in [3, 15, 31]
            ]
        )
        * y_true_1h
        * 0.5
        + 1.0
    )
    omega = ops.max(omega, axis=-1, keepdims=True)

    sample_weight = omega if sample_weight is None else omega * sample_weight
    sample_weight = ops.stop_gradient(sample_weight)

    omega_mean = ops.mean(omega, axis=[1, 2, 3])
    omega_mean = ops.stop_gradient(omega_mean)

    ace = crossentropy(
        y_true,
        y_pred,
        sample_weight,
        from_logits,
        label_smoothing=label_smoothing,
        force_binary=force_binary,
    ) / (omega_mean + 0.5)
    aiou = iou(
        y_true,
        y_pred,
        sample_weight,
        from_logits,
        label_smoothing=label_smoothing,
        force_binary=force_binary,
    )
    amae = mae(
        y_true,
        y_pred,
        sample_weight,
        from_logits,
        regression=False,
        label_smoothing=label_smoothing,
        force_binary=force_binary,
    ) / (
        omega_mean - 0.5
    )  # -1 will produce NaNs

    loss = ace + aiou + amae

    return loss
