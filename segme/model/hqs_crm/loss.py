from segme.loss import CrossEntropyLoss, MeanAbsoluteError, MeanSquaredError, SobelEdgeLoss
from segme.loss.weighted_wrapper import WeightedLossFunctionWrapper


def total_loss(y_true, y_pred, sample_weight=None):
    return CrossEntropyLoss()(y_true, y_pred, sample_weight) + \
           0.5 * MeanAbsoluteError()(y_true, y_pred, sample_weight) + \
           0.5 * MeanSquaredError()(y_true, y_pred, sample_weight) + \
           2.0 * SobelEdgeLoss()(y_true, y_pred, sample_weight)


def hqs_crm_loss():
    return WeightedLossFunctionWrapper(total_loss)
