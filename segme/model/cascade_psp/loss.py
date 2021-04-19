from tensorflow.keras.losses import MeanAbsoluteError, MeanSquaredError, BinaryCrossentropy
from ...loss import SobelEdgeLoss


def _loss_224(y_true, y_pred, sample_weight=None):
    return MeanAbsoluteError()(y_true, y_pred, sample_weight) + \
           MeanSquaredError()(y_true, y_pred, sample_weight) + \
           5. * SobelEdgeLoss()(y_true, y_pred, sample_weight)


def _loss_28(y_true, y_pred, sample_weight=None):
    return BinaryCrossentropy()(y_true, y_pred, sample_weight)


def _loss_56(y_true, y_pred, sample_weight=None):
    return .5 * BinaryCrossentropy()(y_true, y_pred, sample_weight) + \
           .25 * MeanAbsoluteError()(y_true, y_pred, sample_weight) + \
           .25 * MeanSquaredError()(y_true, y_pred, sample_weight)


def _loss_28_2(y_true, y_pred, sample_weight=None):
    return BinaryCrossentropy()(y_true, y_pred, sample_weight)


def _loss_28_3(y_true, y_pred, sample_weight=None):
    return BinaryCrossentropy()(y_true, y_pred, sample_weight)


def _loss_56_2(y_true, y_pred, sample_weight=None):
    return .5 * BinaryCrossentropy()(y_true, y_pred, sample_weight) + \
           .25 * MeanAbsoluteError()(y_true, y_pred, sample_weight) + \
           .25 * MeanSquaredError()(y_true, y_pred, sample_weight)


def cascade_psp_losses():
    return [_loss_224, _loss_56_2, _loss_28_3, _loss_56, _loss_28_2, _loss_28]
