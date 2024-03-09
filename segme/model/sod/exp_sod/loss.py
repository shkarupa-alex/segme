from segme.loss import CrossEntropyLoss, RegionMutualInformationLoss, MeanAbsoluteRegressionError, \
    WeightedLossFunctionWrapper


def reg_loss(y_true, y_pred, sample_weight, loss_weight):
    loss = MeanAbsoluteRegressionError()(y_true, y_pred, sample_weight)
    loss *= loss_weight

    return loss


def cls_loss(y_true, y_pred, sample_weight, loss_weight, with_rmi):
    loss = CrossEntropyLoss()(y_true, y_pred, sample_weight)

    if with_rmi:
        loss += RegionMutualInformationLoss()(y_true, y_pred, sample_weight)

    loss *= loss_weight

    return loss


def exp_sod_losses(backbone_scales, with_depth=False, with_unknown=False):
    weights = [32, 16, 8, 4, 2, 1][:backbone_scales]
    weights = [weights[-1] / w for w in weights]

    losses = []
    for i, w in enumerate(weights):
        with_rmi = i + 1 == backbone_scales
        with_rmi = False  # TODO
        losses.append(lambda y_true, y_pred, sample_weight: cls_loss(y_true, y_pred, sample_weight, w, with_rmi))

        if with_depth:
            losses.append(lambda y_true, y_pred, sample_weight: reg_loss(y_true, y_pred, sample_weight, w * 0.5))

        if with_unknown:
            losses.append(lambda y_true, y_pred, sample_weight: cls_loss(y_true, y_pred, sample_weight, w, False))

    return [WeightedLossFunctionWrapper(l) for l in losses]
