from segme.loss import CrossEntropyLoss, MeanAbsoluteClassificationError, MeanSquaredClassificationError, \
    SobelEdgeLoss, WeightedLossFunctionWrapper


def _loss_224(y_true, y_pred, sample_weight=None):
    return MeanAbsoluteClassificationError()(y_true, y_pred, sample_weight) + \
           MeanSquaredClassificationError()(y_true, y_pred, sample_weight) + \
           5. * SobelEdgeLoss()(y_true, y_pred, sample_weight)


def _loss_28(y_true, y_pred, sample_weight=None):
    return CrossEntropyLoss()(y_true, y_pred, sample_weight)


def _loss_56(y_true, y_pred, sample_weight=None):
    return .5 * CrossEntropyLoss()(y_true, y_pred, sample_weight) + \
           .25 * MeanAbsoluteClassificationError()(y_true, y_pred, sample_weight) + \
           .25 * MeanSquaredClassificationError()(y_true, y_pred, sample_weight)


def _loss_28_2(y_true, y_pred, sample_weight=None):
    return CrossEntropyLoss()(y_true, y_pred, sample_weight)


def _loss_28_3(y_true, y_pred, sample_weight=None):
    return CrossEntropyLoss()(y_true, y_pred, sample_weight)


def _loss_56_2(y_true, y_pred, sample_weight=None):
    return .5 * CrossEntropyLoss()(y_true, y_pred, sample_weight) + \
           .25 * MeanAbsoluteClassificationError()(y_true, y_pred, sample_weight) + \
           .25 * MeanSquaredClassificationError()(y_true, y_pred, sample_weight)


def total_losses():
    return [_loss_224, _loss_56_2, _loss_28_3, _loss_56, _loss_28_2, _loss_28]


def cascade_psp_losses():
    return [WeightedLossFunctionWrapper(tl) for tl in total_losses()]
