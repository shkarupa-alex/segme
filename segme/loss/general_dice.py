import tensorflow as tf
from keras import backend
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.losses_utils import ReductionV2 as Reduction
from .weighted_wrapper import WeightedLossFunctionWrapper


@register_keras_serializable(package='SegMe')
class GeneralizedDiceLoss(WeightedLossFunctionWrapper):
    """ Proposed in: 'Generalised Dice overlap as a deep learning loss function for highly unbalanced segmentations'

    Implements Equations from https://arxiv.org/pdf/1707.03237v3.pdf
    """

    def __init__(self, from_logits=False, reduction=Reduction.AUTO, name='generalized_dice_loss'):
        super().__init__(generalized_dice_loss, reduction=reduction, name=name, from_logits=from_logits)


def generalized_dice_loss(y_true, y_pred, sample_weight, from_logits):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.convert_to_tensor(y_true, 'int32')
    epsilon = tf.convert_to_tensor(backend.epsilon(), dtype=y_pred.dtype)

    if 4 != y_pred.shape.rank or 4 != y_true.shape.rank:
        raise ValueError('Both labels and predictions must have rank 4.')

    if y_pred.shape[-1] is None or y_true.shape[-1] is None:
        raise ValueError('Channel dimension of both labels and predictions must be defined.')

    if 1 != y_true.shape[-1]:
        raise ValueError('Labels must be sparse-encoded.')

    if sample_weight is not None:
        valid_mask = sample_weight > 0.
        valid_weight = tf.cast(valid_mask, y_pred.dtype)

        axis_hwc = list(range(1, sample_weight.shape.ndims))
        batch_weight = tf.math.divide_no_nan(
            tf.reduce_sum(sample_weight, axis=axis_hwc), tf.reduce_sum(valid_weight, axis=axis_hwc))

        y_true = tf.where(valid_mask, y_true, tf.zeros_like(y_true))
        y_pred = tf.where(valid_mask, y_pred, tf.zeros_like(y_pred) + epsilon)
    else:
        batch_weight = None
        valid_weight = None

    if from_logits:
        if 1 == y_pred.shape[-1]:
            y_pred = tf.nn.sigmoid(y_pred)
        else:
            y_pred = tf.nn.softmax(y_pred)

    if 1 == y_pred.shape[-1]:
        y_pred = tf.concat([1. - y_pred, y_pred], axis=-1)

    y_true_1h = tf.one_hot(y_true[..., 0], y_pred.shape[-1])
    y_true_1h = tf.cast(y_true_1h, dtype=y_pred.dtype)

    weight = y_true_1h if valid_weight is None else y_true_1h * valid_weight
    weight = tf.math.divide_no_nan(1., tf.reduce_sum(weight, axis=[1, 2]) ** 2)
    weight = tf.stop_gradient(weight)

    intersection = y_pred * y_true_1h
    intersection = intersection if valid_weight is None else intersection * valid_weight
    intersection = tf.reduce_sum(intersection, axis=[1, 2]) * weight
    intersection = tf.reduce_sum(intersection, axis=-1)

    union = y_pred + y_true_1h
    union = union if valid_weight is None else union * valid_weight
    union = tf.reduce_sum(union, axis=[1, 2]) * weight
    union = tf.reduce_sum(union, axis=-1)

    loss = 1. - 2. * intersection / (union + epsilon)
    if batch_weight is not None:
        loss *= batch_weight

    return loss
