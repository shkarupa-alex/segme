from keras.src import ops
from keras.src.saving import register_keras_serializable

from segme.loss.common_loss import compute_gradient
from segme.loss.common_loss import validate_input
from segme.loss.common_loss import weighted_loss
from segme.loss.weighted_wrapper import WeightedLossFunctionWrapper


@register_keras_serializable(package="SegMe>Loss")
class ReflectionTransmissionExclusionLoss(WeightedLossFunctionWrapper):
    """Proposed in: 'Single Image Reflection Removal with Perceptual Losses'

    Implements Equation [5] in https://arxiv.org/pdf/1806.05376
    """

    def __init__(
        self,
        levels=3,
        reduction="sum_over_batch_size",
        name="reflection_transmission_exclusion_loss",
    ):
        super().__init__(
            reflection_transmission_exclusion_loss,
            reduction=reduction,
            name=name,
            levels=levels,
        )


def _exclusion_level(r_pred, t_pred, axis, sample_weight):
    grad_r = compute_gradient(r_pred, axis, "sub")
    grad_t = compute_gradient(t_pred, axis, "sub")
    grad_w = (
        None
        if sample_weight is None
        else compute_gradient(sample_weight, axis, "min") ** 4
    )

    alpha = 2.0 * ops.divide_no_nan(
        ops.mean(ops.abs(grad_r), axis=[1, 2, 3], keepdims=True),
        ops.mean(ops.abs(grad_t), axis=[1, 2, 3], keepdims=True),
    )
    grad_rs = ops.sigmoid(grad_r) * 2.0 - 1.0
    grad_ts = ops.sigmoid(grad_t * alpha) * 2.0 - 1.0

    loss = ops.square(grad_rs) * ops.square(grad_ts)
    loss = weighted_loss(loss, grad_w) ** 0.25

    return loss


def _down_sample(reflections, transmissions, weights):
    height, width = ops.shape(reflections)[1:3]
    hpad, wpad = height % 2, width % 2
    paddings = [[0, 0], [0, hpad], [0, wpad], [0, 0]]

    reflections = ops.pad(reflections, paddings, "SYMMETRIC")
    transmissions = ops.pad(transmissions, paddings, "SYMMETRIC")
    if weights is not None:
        weights = ops.pad(weights, paddings, "SYMMETRIC")

    reflections = ops.average_pool(reflections, 2, strides=2, padding="valid")
    transmissions = ops.average_pool(
        transmissions, 2, strides=2, padding="valid"
    )
    if weights is not None:
        weights = ops.average_pool(weights, 2, strides=2, padding="valid")
        weights = ops.stop_gradient(weights)

    return reflections, transmissions, weights


def reflection_transmission_exclusion_loss(
    r_pred, t_pred, sample_weight, levels
):
    r_pred, t_pred, sample_weight = validate_input(
        r_pred, t_pred, sample_weight, dtype=None, rank=4, channel="same"
    )

    loss = []
    for level in range(levels):
        loss.append(
            _exclusion_level(
                r_pred, t_pred, axis=1, sample_weight=sample_weight
            )
        )
        loss.append(
            _exclusion_level(
                r_pred, t_pred, axis=2, sample_weight=sample_weight
            )
        )
        last_level = levels - 1 == level

        if not last_level:
            height, width = ops.shape(r_pred)[1:3]
            if isinstance(height, int) and isinstance(width, int):
                if height <= 2 or width <= 2:
                    raise ValueError(
                        "Reflection-transmission loss got inputs with "
                        "spatial size <= 2 at some pyramid level."
                    )
            r_pred, t_pred, sample_weight = _down_sample(
                r_pred, t_pred, sample_weight
            )

    loss = sum(loss) / (2.0 * levels)

    return loss
