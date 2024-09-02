from keras.src import backend
from keras.src import ops
from keras.src.saving import register_keras_serializable

from segme.loss.common_loss import validate_input
from segme.loss.weighted_wrapper import WeightedLossFunctionWrapper


@register_keras_serializable(package="SegMe>Loss")
class StrongerTeacherLoss(WeightedLossFunctionWrapper):
    """Proposed in: 'Knowledge Distillation from A Stronger Teacher'

    Implements equations [8] and [9] in https://arxiv.org/pdf/2205.10536
    """

    def __init__(
        self,
        temperature=1.0,
        reduction="sum_over_batch_size",
        name="stronger_teacher_loss",
    ):
        super().__init__(
            stronger_teacher_loss,
            reduction=reduction,
            name=name,
            temperature=temperature,
        )


def stronger_teacher_loss(y_true, y_pred, sample_weight, temperature):
    y_true, y_pred, sample_weight = validate_input(
        y_true, y_pred, sample_weight, dtype=None, rank=None, channel="same"
    )

    if (
        sample_weight is not None
        and sample_weight.shape.rank != y_true.shape.rank
    ):
        raise ValueError(
            "Sample weights and `y_true`/`y_true` ranks must be equal."
        )

    inv_temperature = 1.0 / temperature
    y_true = ops.softmax(y_true * inv_temperature, axis=-1)
    y_true = ops.stop_gradient(y_true)
    y_pred = ops.softmax(y_pred * inv_temperature, axis=-1)

    # Inter-class loss is not sensitive to sample weights
    loss = _inter_class_relation(y_true, y_pred, None) + _intra_class_relation(
        y_true, y_pred, sample_weight
    )
    loss *= temperature**2

    return loss


def _weighted_average(x, weights=None):
    if weights is None:
        return ops.mean(x, axis=-1)

    if "float16" == backend.standardize_dtype(weights.dtype):
        epsilon = 2.4e-10
    else:
        epsilon = 1e-12

    weighted_sum = ops.sum(x * weights, axis=-1)
    weights_sum = ops.maximum(ops.sum(weights, axis=-1), epsilon)

    return weighted_sum / weights_sum


def _cosine_similarity(u, v, weights=None):
    uv = _weighted_average(u * v, weights)
    uu = _weighted_average(ops.square(u), weights)
    vv = _weighted_average(ops.square(v), weights)

    if "float16" in {
        backend.standardize_dtype(u.dtype),
        backend.standardize_dtype(v.dtype),
    }:
        epsilon = 1.55e-5
    else:
        epsilon = 1e-12

    inv_norm = ops.rsqrt(ops.maximum(uu * vv, epsilon))

    return -uv * inv_norm


def _pearson_correlation(a, b, weights=None):
    a = a - ops.mean(a, axis=-1, keepdims=True)
    b = b - ops.mean(b, axis=-1, keepdims=True)

    return _cosine_similarity(a, b, weights)


def _inter_class_relation(y_true, y_pred, sample_weight=None):
    return ops.mean(_pearson_correlation(y_true, y_pred, sample_weight))


def _intra_class_relation(y_true, y_pred, sample_weight):
    channels = y_true.shape[-1]

    y_true = ops.reshape(y_true, [-1, channels])
    y_true = ops.transpose(y_true, [1, 0])

    y_pred = ops.reshape(y_pred, [-1, channels])
    y_pred = ops.transpose(y_pred, [1, 0])

    if sample_weight is not None:
        sample_weight = ops.reshape(sample_weight, [1, -1])

    return _inter_class_relation(y_true, y_pred, sample_weight)
