from tensorflow.keras.losses import BinaryCrossentropy
from ...loss import ConsistencyEnhancedSigmoidLoss


def minet_loss(y_true, y_pred, sample_weight=None):
    return BinaryCrossentropy()(y_true, y_pred, sample_weight) + \
           ConsistencyEnhancedSigmoidLoss()(y_true, y_pred, sample_weight)
