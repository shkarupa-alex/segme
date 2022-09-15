import tensorflow as tf
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.losses_utils import ReductionV2 as Reduction
from segme.loss.common_loss import validate_input, mse
from segme.loss.weighted_wrapper import WeightedLossFunctionWrapper


@register_keras_serializable(package='SegMe>Loss')
class MeanSquaredError(WeightedLossFunctionWrapper):
    def __init__(self, from_logits=False, reduction=Reduction.AUTO, name='mean_squared_error'):
        super().__init__(mean_squared_error, reduction=reduction, name=name, from_logits=from_logits)


def mean_squared_error(y_true, y_pred, sample_weight, from_logits):
    y_true, y_pred, sample_weight = validate_input(
        y_true, y_pred, sample_weight, dtype='int32', rank=4, channel='sparse')

    return mse(y_true, y_pred, sample_weight, from_logits)
