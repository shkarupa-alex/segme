from segme.loss import CrossEntropyLoss
from segme.loss import MeanAbsoluteClassificationError
from segme.loss import MeanSquaredClassificationError
from segme.loss import SobelEdgeLoss
from segme.loss import WeightedLossFunctionWrapper


def _total_loss(y_true, y_pred, sample_weight=None):
    return (
        CrossEntropyLoss()(y_true, y_pred, sample_weight)
        + 0.5 * MeanAbsoluteClassificationError()(y_true, y_pred, sample_weight)
        + 0.5 * MeanSquaredClassificationError()(y_true, y_pred, sample_weight)
        + 2.0 * SobelEdgeLoss()(y_true, y_pred, sample_weight)
    )


def hqs_crm_loss():
    return WeightedLossFunctionWrapper(_total_loss)
