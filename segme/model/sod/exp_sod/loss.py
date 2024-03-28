from segme.loss import AdaptivePixelIntensityLoss, CrossEntropyLoss, RegionMutualInformationLoss


def exp_sod_losses(backbone_scales):
    return [CrossEntropyLoss(label_smoothing=0.1)] + [CrossEntropyLoss() for i in range(backbone_scales - 2)] + [[
        AdaptivePixelIntensityLoss(), RegionMutualInformationLoss()
    ]]
