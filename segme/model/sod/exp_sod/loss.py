from segme.loss import CrossEntropyLoss
from segme.loss import WeightedLossFunctionWrapper
from segme.loss.adaptive_intensity import adaptive_pixel_intensity_loss
from segme.loss.mean_absolute import mean_absolute_regression_error
from segme.loss.region_mutual import region_mutual_information_loss


def ape_rmi(y_true, y_pred, sample_weight):
    return adaptive_pixel_intensity_loss(
        y_true, y_pred, sample_weight, False, 0.0, False
    ) + region_mutual_information_loss(
        y_true, y_pred, sample_weight, 3, 4, "avgpool", False, 0.0, False
    )


def scaled_mae(y_true, y_pred, sample_weight, scale):
    return mean_absolute_regression_error(y_true, y_pred, sample_weight) * scale


def exp_sod_losses(backbone_scales, with_trimap=False):
    losses = [CrossEntropyLoss(label_smoothing=0.1)]
    if with_trimap:
        losses.append(CrossEntropyLoss(label_smoothing=0.1))

    losses = []
    for i in range(backbone_scales):
        label_smoothing = 0.1 if 0 == i else 0.0

        if backbone_scales - 1 == i:
            losses.append(WeightedLossFunctionWrapper(ape_rmi))
        else:
            losses.append(CrossEntropyLoss(label_smoothing=label_smoothing))

        if with_trimap:
            losses.append(CrossEntropyLoss(label_smoothing=label_smoothing))

    return losses
