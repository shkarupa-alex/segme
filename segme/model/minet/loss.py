from keras.losses import BinaryCrossentropy
from ...loss import ConsistencyEnhancedSigmoidLoss, WeightedLossFunctionWrapper


def total_loss(y_true, y_pred, sample_weight=None):
    return BinaryCrossentropy()(y_true, y_pred, sample_weight) + \
           ConsistencyEnhancedSigmoidLoss()(y_true, y_pred, sample_weight)


def minet_loss():
    return WeightedLossFunctionWrapper(total_loss)
