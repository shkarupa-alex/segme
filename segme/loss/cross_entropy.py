import tensorflow as tf
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.losses_utils import ReductionV2 as Reduction
from segme.loss.common_loss import validate_input, crossentropy
from segme.loss.weighted_wrapper import WeightedLossFunctionWrapper


@register_keras_serializable(package='SegMe>Loss')
class CrossEntropyLoss(WeightedLossFunctionWrapper):
    def __init__(self, from_logits=False, reduction=Reduction.AUTO, name='cross_entropy_loss'):
        super().__init__(cross_entropy_loss, reduction=reduction, name=name, from_logits=from_logits)


def cross_entropy_loss(y_true, y_pred, sample_weight, from_logits):
    y_true, y_pred, sample_weight = validate_input(
        y_true, y_pred, sample_weight, dtype='int32', rank=4, channel='sparse')

    return crossentropy(y_true, y_pred, sample_weight, from_logits)
