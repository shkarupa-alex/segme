from ...loss import AdaptivePixelIntensityLoss


def tracer_losses():
    return [AdaptivePixelIntensityLoss() for _ in range(5)]
