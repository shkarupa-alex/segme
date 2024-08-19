from segme.loss import CrossEntropyLoss
from segme.loss import SobelEdgeLoss
from segme.loss import WeightedLossFunctionWrapper


def total_loss(y_true, y_pred, sample_weight=None):
    return CrossEntropyLoss()(
        y_true, y_pred, sample_weight
    ) + 5.0 * SobelEdgeLoss()(y_true, y_pred, sample_weight)


def seg_refiner_loss():
    return WeightedLossFunctionWrapper(total_loss)
