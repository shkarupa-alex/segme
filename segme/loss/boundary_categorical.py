import tensorflow as tf
from keras.src.saving import register_keras_serializable
from tfmiss.image import euclidean_distance

from segme.loss.common_loss import to_1hot
from segme.loss.common_loss import to_probs
from segme.loss.common_loss import validate_input
from segme.loss.common_loss import weighted_loss
from segme.loss.weighted_wrapper import WeightedLossFunctionWrapper


@register_keras_serializable(package="SegMe>Loss")
class BoundaryCategoricalLoss(WeightedLossFunctionWrapper):
    """Proposed in: 'Boundary loss for highly unbalanced segmentation'

    Implements Equation (5) from https://arxiv.org/pdf/1812.07032v4.pdf
    """

    def __init__(
        self,
        from_logits=False,
        reduction="sum_over_batch_size",
        name="boundary_categorical_loss",
    ):
        super().__init__(
            boundary_categorical_loss,
            reduction=reduction,
            name=name,
            from_logits=from_logits,
        )


def boundary_categorical_loss(y_true, y_pred, sample_weight, from_logits):
    y_true, y_pred, sample_weight = validate_input(
        y_true, y_pred, sample_weight, dtype="uint8", rank=4, channel="sparse"
    )
    y_pred, from_logits = to_probs(y_pred, from_logits, force_binary=True)
    y_true, y_pred = to_1hot(y_true, y_pred, from_logits)
    y_false = 1 - y_true

    has_true = tf.reduce_any(y_true == 1, axis=[1, 2], keepdims=True)
    has_false = tf.reduce_any(y_true == 0, axis=[1, 2], keepdims=True)

    d_true = tf.cast(euclidean_distance(y_true), y_pred.dtype)
    d_false = tf.cast(euclidean_distance(y_false), y_pred.dtype)

    distance = d_false * tf.cast(y_false, dtype=y_pred.dtype) - (
        d_true - 1.0
    ) * tf.cast(y_true, dtype=y_pred.dtype)
    distance *= tf.cast(has_true & has_false, "float32")
    distance = tf.stop_gradient(distance)

    loss = weighted_loss(y_pred * distance, sample_weight)

    return loss
