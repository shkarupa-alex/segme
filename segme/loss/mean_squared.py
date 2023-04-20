from keras.saving import register_keras_serializable
from keras.src.utils.losses_utils import ReductionV2 as Reduction
from segme.loss.common_loss import validate_input, mse
from segme.loss.weighted_wrapper import WeightedLossFunctionWrapper


@register_keras_serializable(package='SegMe>Loss')
class MeanSquaredClassificationError(WeightedLossFunctionWrapper):
    def __init__(self, from_logits=False, reduction=Reduction.AUTO, name='mean_squared_classification_error'):
        super().__init__(mean_squared_classification_error, reduction=reduction, name=name, from_logits=from_logits)


@register_keras_serializable(package='SegMe>Loss')
class MeanSquaredRegressionError(WeightedLossFunctionWrapper):
    def __init__(self, reduction=Reduction.AUTO, name='mean_squared_regression_error'):
        super().__init__(mean_squared_regression_error, reduction=reduction, name=name)


def mean_squared_classification_error(y_true, y_pred, sample_weight, from_logits):
    y_true, y_pred, sample_weight = validate_input(
        y_true, y_pred, sample_weight, dtype='int32', rank=4, channel='sparse')

    return mse(y_true, y_pred, sample_weight, from_logits)


def mean_squared_regression_error(y_true, y_pred, sample_weight):
    y_true, y_pred, sample_weight = validate_input(
        y_true, y_pred, sample_weight, dtype=None, rank=4, channel='same')

    return mse(y_true, y_pred, sample_weight, from_logits=False, regression=True)
