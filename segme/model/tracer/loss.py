from ...loss import BinaryAdaptivePixelIntensityLoss


def tracer_losses():
    return [BinaryAdaptivePixelIntensityLoss() for _ in range(5)]
