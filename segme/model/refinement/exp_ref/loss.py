from segme.loss import CrossEntropyLoss, MeanAbsoluteClassificationError, MeanSquaredClassificationError, \
    SobelEdgeLoss, WeightedLossFunctionWrapper


def _loss_8(y_true, y_pred, sample_weight=None):
    return CrossEntropyLoss()(y_true, y_pred, sample_weight)


def _loss_4(y_true, y_pred, sample_weight=None):
    return 0.5 * CrossEntropyLoss()(y_true, y_pred, sample_weight) + \
           0.5 * MeanAbsoluteClassificationError()(y_true, y_pred, sample_weight) + \
           0.5 * MeanSquaredClassificationError()(y_true, y_pred, sample_weight)


def _loss_2(y_true, y_pred, sample_weight=None):
    return 1.0 * MeanAbsoluteClassificationError()(y_true, y_pred, sample_weight) + \
           0.5 * MeanSquaredClassificationError()(y_true, y_pred, sample_weight) + \
           2.0 * SobelEdgeLoss()(y_true, y_pred, sample_weight)


def total_losses():
    return [_loss_2, _loss_4, _loss_8]


def exp_ref_losses():
    return [WeightedLossFunctionWrapper(tl) for tl in total_losses()]