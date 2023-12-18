import tensorflow as tf
from keras.saving import register_keras_serializable
from keras.src.utils.losses_utils import ReductionV2 as Reduction
from segme.loss.common_loss import validate_input, weighted_loss
from segme.loss.weighted_wrapper import WeightedLossFunctionWrapper


@register_keras_serializable(package='SegMe>Loss')
class KLDivergenceLoss(WeightedLossFunctionWrapper):
    def __init__(self, temperature=1., reduction=Reduction.AUTO, name='kl_divergence_loss'):
        super().__init__(kl_divergence_loss, reduction=reduction, name=name, temperature=temperature)


def kl_divergence_loss(y_true, y_pred, sample_weight, temperature):
    y_true, y_pred, sample_weight = validate_input(
        y_true, y_pred, sample_weight, dtype=None, rank=None, channel='same')

    inv_temperature = 1. / temperature
    loss = (temperature ** 2) * tf.nn.softmax(y_true) * (
            tf.nn.log_softmax(y_true * inv_temperature) - tf.nn.log_softmax(y_pred * inv_temperature))
    loss = tf.reduce_sum(loss, axis=-1, keepdims=True)

    return weighted_loss(loss, sample_weight)
