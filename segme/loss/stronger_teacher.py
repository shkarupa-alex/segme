import tensorflow as tf
from keras.saving import register_keras_serializable
from keras.src.utils.losses_utils import ReductionV2 as Reduction
from segme.loss.common_loss import validate_input
from segme.loss.weighted_wrapper import WeightedLossFunctionWrapper


@register_keras_serializable(package='SegMe>Loss')
class StrongerTeacherLoss(WeightedLossFunctionWrapper):
    """ Proposed in: 'Knowledge Distillation from A Stronger Teacher'

    Implements equations [8] and [9] in https://arxiv.org/pdf/2205.10536.pdf
    """

    def __init__(self, temperature=1., reduction=Reduction.AUTO, name='stronger_teacher_loss'):
        super().__init__(stronger_teacher_loss, reduction=reduction, name=name, temperature=temperature)


def stronger_teacher_loss(y_true, y_pred, sample_weight, temperature):
    y_true, y_pred, sample_weight = validate_input(
        y_true, y_pred, sample_weight, dtype=None, rank=None, channel='same')

    if sample_weight is not None and sample_weight.shape.rank != y_true.shape.rank:
        raise ValueError('Sample weights and `y_true`/`y_true` ranks must be equal.')

    inv_temperature = 1. / temperature
    y_true = tf.nn.softmax(y_true * inv_temperature, axis=-1)
    y_pred = tf.nn.softmax(y_pred * inv_temperature, axis=-1)

    # Inter-class loss is not sensitive to sample weights
    loss = _inter_class_relation(y_true, y_pred, None) + _intra_class_relation(y_true, y_pred, sample_weight)
    loss *= temperature ** 2

    return loss


def _weighted_average(x, weights=None):
    if weights is None:
        return tf.reduce_mean(x, axis=-1)

    epsilon = 2.4e-10 if tf.float16 == weights.dtype else 1e-12
    weighted_sum = tf.reduce_sum(x * weights, axis=-1)
    weights_sum = tf.maximum(tf.reduce_sum(weights, axis=-1), epsilon)

    return weighted_sum / weights_sum


def _cosine_similarity(u, v, weights=None):
    uv = _weighted_average(u * v, weights)
    uu = _weighted_average(tf.square(u), weights)
    vv = _weighted_average(tf.square(v), weights)

    epsilon = 1.55e-5 if tf.float16 in {u.dtype, v.dtype} else 1e-12
    inv_norm = tf.math.rsqrt(tf.maximum(uu * vv, epsilon))

    return - uv * inv_norm


def _pearson_correlation(a, b, weights=None):
    a = a - tf.reduce_mean(a, axis=-1, keepdims=True)
    b = b - tf.reduce_mean(b, axis=-1, keepdims=True)

    return _cosine_similarity(a, b, weights)


def _inter_class_relation(y_true, y_pred, sample_weight=None):
    return tf.reduce_mean(_pearson_correlation(y_true, y_pred, sample_weight))


def _intra_class_relation(y_true, y_pred, sample_weight):
    channels = y_true.shape[-1]

    y_true = tf.reshape(y_true, [-1, channels])
    y_true = tf.transpose(y_true, [1, 0])

    y_pred = tf.reshape(y_pred, [-1, channels])
    y_pred = tf.transpose(y_pred, [1, 0])

    if sample_weight is not None:
        sample_weight = tf.reshape(sample_weight, [1, -1])

    return _inter_class_relation(y_true, y_pred, sample_weight)
