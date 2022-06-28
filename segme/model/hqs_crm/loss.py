from keras.losses import MeanAbsoluteError, MeanSquaredError, BinaryCrossentropy, LossFunctionWrapper
from ...loss import SobelEdgeLoss


def total_loss(y_true, y_pred, sample_weight=None):
    return BinaryCrossentropy()(y_true, y_pred, sample_weight) + \
           0.5 * MeanAbsoluteError()(y_true, y_pred, sample_weight) + \
           0.5 * MeanSquaredError()(y_true, y_pred, sample_weight) + \
           2.0 * SobelEdgeLoss()(y_true, y_pred, sample_weight)


def hqs_crm_loss():
    return LossFunctionWrapper(total_loss)
