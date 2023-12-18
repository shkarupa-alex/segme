from keras.saving import register_keras_serializable
from keras.src.utils.losses_utils import ReductionV2 as Reduction
from segme.loss.common_loss import validate_input, crossentropy
from segme.loss.weighted_wrapper import WeightedLossFunctionWrapper


@register_keras_serializable(package='SegMe>Loss')
class CrossEntropyLoss(WeightedLossFunctionWrapper):
    def __init__(
            self, from_logits=False, force_binary=False, label_smoothing=0., reduction=Reduction.AUTO,
            name='cross_entropy_loss'):
        super().__init__(
            cross_entropy_loss, reduction=reduction, name=name, from_logits=from_logits,
            force_binary=force_binary, label_smoothing=label_smoothing)


def cross_entropy_loss(y_true, y_pred, sample_weight, from_logits, force_binary, label_smoothing):
    y_true, y_pred, sample_weight = validate_input(
        y_true, y_pred, sample_weight, dtype=None, rank=None, channel=None)

    return crossentropy(y_true, y_pred, sample_weight, from_logits, force_binary, label_smoothing)
