from segme.loss import AdaptivePixelIntensityLoss, LaplaceEdgeCrossEntropy, MeanAbsoluteClassificationError, \
    MeanSquaredClassificationError, SobelEdgeLoss
from segme.loss import CrossEntropyLoss, RegionMutualInformationLoss, MeanAbsoluteRegressionError, \
    WeightedLossFunctionWrapper


def cls_loss(y_true, y_pred, sample_weight, scale_weight, loss_weights):
    assert 6 == len(loss_weights)
    loss_weights = [w / sum(loss_weights) for w in loss_weights]

    loss = [
        CrossEntropyLoss()(y_true, y_pred, sample_weight),
        AdaptivePixelIntensityLoss()(y_true, y_pred, sample_weight),
        MeanAbsoluteClassificationError()(y_true, y_pred, sample_weight),
        MeanSquaredClassificationError()(y_true, y_pred, sample_weight),
        LaplaceEdgeCrossEntropy()(y_true, y_pred, sample_weight),
        SobelEdgeLoss()(y_true, y_pred, sample_weight)
    ]

    loss = sum([l * w for l, w in zip(loss, loss_weights)]) * scale_weight

    return loss


def exp_sod_losses(backbone_scales, scale_weights, loss_weights):
    assert len(loss_weights) == backbone_scales
    scale_weights = [w / sum(scale_weights) for w in scale_weights]

    losses = []
    for i, w in enumerate(scale_weights):
        losses.append(
            lambda y_true, y_pred, sample_weight: cls_loss(y_true, y_pred, sample_weight, w, loss_weights[i]))

    return [WeightedLossFunctionWrapper(l) for l in losses]
