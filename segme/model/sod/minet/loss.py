from segme.loss import CrossEntropyLoss, ConsistencyEnhancedLoss, WeightedLossFunctionWrapper


def total_loss(y_true, y_pred, sample_weight=None):
    return CrossEntropyLoss()(y_true, y_pred, sample_weight) + \
           ConsistencyEnhancedLoss()(y_true, y_pred, sample_weight)


def minet_loss():
    return WeightedLossFunctionWrapper(total_loss)
