from segme.loss import CrossEntropyLoss, MeanAbsoluteClassificationError, MeanSquaredClassificationError, \
    SobelEdgeLoss, WeightedLossFunctionWrapper


def total_loss(y_true, y_pred, sample_weight=None):
    return CrossEntropyLoss()(y_true, y_pred, sample_weight) + \
           0.5 * MeanAbsoluteClassificationError()(y_true, y_pred, sample_weight) + \
           0.5 * MeanSquaredClassificationError()(y_true, y_pred, sample_weight) + \
           2.0 * SobelEdgeLoss()(y_true, y_pred, sample_weight)


def hqs_crm_loss():
    return WeightedLossFunctionWrapper(total_loss)
