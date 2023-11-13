import tensorflow as tf
from keras.saving import register_keras_serializable
from keras.src.utils.losses_utils import ReductionV2 as Reduction
from segme.loss.common_loss import validate_input, weighted_loss
from segme.loss.weighted_wrapper import WeightedLossFunctionWrapper


@register_keras_serializable(package='SegMe>Loss')
class SoftMeanAbsoluteError(WeightedLossFunctionWrapper):
    def __init__(self, beta=1., reduction=Reduction.AUTO, name='soft_mean_absolute_error'):
        super().__init__(soft_mean_absolute_error, reduction=reduction, name=name, beta=beta)


def soft_mean_absolute_error(y_true, y_pred, sample_weight, beta):
    y_true, y_pred, sample_weight = validate_input(
        y_true, y_pred, sample_weight, dtype=None, rank=None, channel='same')
    beta = tf.cast(beta, dtype=y_pred.dtype)

    error = y_pred - y_true
    abs_error = tf.abs(error)
    square_error = tf.square(error)

    loss = tf.where(abs_error < beta, square_error * (0.5 / beta), abs_error - 0.5 * beta)
    loss = weighted_loss(loss, sample_weight)

    return loss
