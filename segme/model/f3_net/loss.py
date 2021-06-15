import tensorflow as tf
from ...loss import PixelPositionAwareLoss


class ScaledPixelPositionAwareLoss(PixelPositionAwareLoss):
    def __init__(
            self, from_logits=False, gamma=5, ksize=31, reduction=tf.keras.losses.Reduction.AUTO,
            name='pixel_position_aware_loss', scale=1.):
        super().__init__(reduction=reduction, name=name, from_logits=from_logits, gamma=gamma, ksize=ksize)
        self.scale = scale

    def get_config(self):
        config = super().get_config()
        config.update({'scale': self.scale})

        return config

    def __call__(self, y_true, y_pred, sample_weight=None):
        return super().__call__(y_true, y_pred, sample_weight) * self.scale


def f3net_losses():
    return [
        PixelPositionAwareLoss(),
        PixelPositionAwareLoss(),
        PixelPositionAwareLoss(),
        ScaledPixelPositionAwareLoss(scale=1 / 2),
        ScaledPixelPositionAwareLoss(scale=1 / 4),
        ScaledPixelPositionAwareLoss(scale=1 / 8)
    ]
