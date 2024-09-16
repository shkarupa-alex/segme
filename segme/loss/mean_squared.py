from keras.src.saving import register_keras_serializable

from segme.loss.common_loss import mse
from segme.loss.common_loss import validate_input
from segme.loss.weighted_wrapper import WeightedLossFunctionWrapper


@register_keras_serializable(package="SegMe>Loss")
class MeanSquaredClassificationError(WeightedLossFunctionWrapper):
    def __init__(
        self,
        from_logits=False,
        label_smoothing=0.0,
        reduction="sum_over_batch_size",
        name="mean_squared_classification_error",
    ):
        super().__init__(
            mean_squared_classification_error,
            reduction=reduction,
            name=name,
            from_logits=from_logits,
            label_smoothing=label_smoothing,
        )


@register_keras_serializable(package="SegMe>Loss")
class MeanSquaredRegressionError(WeightedLossFunctionWrapper):
    def __init__(
        self,
        reduction="sum_over_batch_size",
        name="mean_squared_regression_error",
    ):
        super().__init__(
            mean_squared_regression_error, reduction=reduction, name=name
        )


def mean_squared_classification_error(
    y_true, y_pred, sample_weight, from_logits, label_smoothing
):
    y_true, y_pred, sample_weight = validate_input(
        y_true, y_pred, sample_weight, dtype="int64", rank=4, channel="sparse"
    )

    return mse(
        y_true,
        y_pred,
        sample_weight,
        from_logits,
        regression=False,
        label_smoothing=label_smoothing,
    )


def mean_squared_regression_error(y_true, y_pred, sample_weight):
    y_true, y_pred, sample_weight = validate_input(
        y_true, y_pred, sample_weight, dtype=None, rank=4, channel="same"
    )

    return mse(
        y_true, y_pred, sample_weight, from_logits=False, regression=True
    )
