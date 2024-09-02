import numpy as np
from keras.src import ops
from keras.src.saving import register_keras_serializable

from segme.loss.common_loss import to_1hot
from segme.loss.common_loss import to_probs
from segme.loss.common_loss import validate_input
from segme.loss.common_loss import weighted_loss
from segme.loss.weighted_wrapper import WeightedLossFunctionWrapper


@register_keras_serializable(package="SegMe>Loss")
class LaplaceEdgeCrossEntropy(WeightedLossFunctionWrapper):
    """Proposed in: 'Pyramid Feature Attention Network for Saliency detection
    (2019)'

    Implements Equation [10] in https://arxiv.org/pdf/1903.00179
    Compute edge loss with Laplace operator
    """

    def __init__(
        self,
        from_logits=False,
        force_binary=False,
        reduction="sum_over_batch_size",
        name="laplace_edge_cross_entropy",
    ):
        super().__init__(
            laplace_edge_cross_entropy,
            reduction=reduction,
            name=name,
            from_logits=from_logits,
            force_binary=force_binary,
        )


def laplace(probs, kernel):
    edge = ops.pad(probs, [(0, 0), (1, 1), (1, 1), (0, 0)], mode="SYMMETRIC")
    edge = ops.depthwise_conv(edge, kernel, strides=1, padding="valid")
    edge = ops.relu(ops.tanh(edge))

    return edge


def laplace_edge_cross_entropy(
    y_true, y_pred, sample_weight, from_logits, force_binary
):
    y_true, y_pred, sample_weight = validate_input(
        y_true, y_pred, sample_weight, dtype="int64", rank=4, channel="sparse"
    )
    y_prob, from_logits = to_probs(
        y_pred, from_logits, force_binary=force_binary
    )
    y_true_1h, y_prob_1h = to_1hot(
        y_true, y_prob, from_logits, dtype=y_prob.dtype
    )

    kernel = np.reshape([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], [3, 3, 1, 1])
    kernel = np.tile(kernel, [1, 1, y_true_1h.shape[-1], 1])
    kernel = ops.cast(kernel, y_prob.dtype)
    kernel = ops.stop_gradient(kernel)

    y_true_edge = laplace(y_true_1h, kernel)[..., None]
    y_true_edge = ops.stop_gradient(y_true_edge)
    y_pred_edge = laplace(y_prob_1h, kernel)[..., None]

    loss = ops.binary_crossentropy(y_true_edge, y_pred_edge, from_logits=False)
    loss = ops.mean(loss, axis=-1)
    loss = weighted_loss(loss, sample_weight)

    return loss
