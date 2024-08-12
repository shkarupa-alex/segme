import tensorflow as tf
from keras.src.saving import register_keras_serializable

from segme.common.shape import get_shape
from segme.loss.common_loss import compute_gradient
from segme.loss.common_loss import validate_input
from segme.loss.common_loss import weighted_loss
from segme.loss.weighted_wrapper import WeightedLossFunctionWrapper


@register_keras_serializable(package="SegMe>Loss")
class ReflectionTransmissionExclusionLoss(WeightedLossFunctionWrapper):
    """Proposed in: 'Single Image Reflection Removal with Perceptual Losses'

    Implements Equation [5] in https://arxiv.org/pdf/1806.05376.pdf
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

    alpha = 2.0 * tf.math.divide_no_nan(
        tf.reduce_mean(tf.abs(grad_r), axis=[1, 2, 3], keepdims=True),
        tf.reduce_mean(tf.abs(grad_t), axis=[1, 2, 3], keepdims=True),
    )
    grad_rs = tf.nn.sigmoid(grad_r) * 2.0 - 1.0
    grad_ts = tf.nn.sigmoid(grad_t * alpha) * 2.0 - 1.0

    loss = tf.square(grad_rs) * tf.square(grad_ts)
    loss = weighted_loss(loss, grad_w) ** 0.25

    return loss


def _down_sample(reflections, transmissions, weights):
    (height, width), _ = get_shape(reflections, axis=[1, 2])
    hpad, wpad = height % 2, width % 2
    paddings = [[0, 0], [0, hpad], [0, wpad], [0, 0]]

    reflections = tf.pad(reflections, paddings, "SYMMETRIC")
    transmissions = tf.pad(transmissions, paddings, "SYMMETRIC")
    if weights is not None:
        weights = tf.pad(weights, paddings, "SYMMETRIC")

    reflections = tf.nn.avg_pool(
        reflections, ksize=2, strides=2, padding="VALID"
    )
    transmissions = tf.nn.avg_pool(
        transmissions, ksize=2, strides=2, padding="VALID"
    )
    if weights is not None:
        weights = tf.nn.avg_pool(weights, ksize=2, strides=2, padding="VALID")
        weights = tf.stop_gradient(weights)

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
            r_size, static_size = get_shape(r_pred, axis=[1, 2])
            if static_size:
                r_size = min(r_size)
            else:
                r_size = tf.minimum(*r_size)
            assert_true_shape = tf.assert_greater(r_size, 2)
            with tf.control_dependencies([assert_true_shape]):
                r_pred, t_pred, sample_weight = _down_sample(
                    r_pred, t_pred, sample_weight
                )
    loss = sum(loss) / (2.0 * levels)

    return loss
