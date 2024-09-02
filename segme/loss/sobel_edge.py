import numpy as np
from keras.src import backend
from keras.src import ops
from keras.src.saving import register_keras_serializable

from segme.loss.common_loss import to_1hot
from segme.loss.common_loss import to_probs
from segme.loss.common_loss import validate_input
from segme.loss.common_loss import weighted_loss
from segme.loss.weighted_wrapper import WeightedLossFunctionWrapper


@register_keras_serializable(package="SegMe>Loss")
class SobelEdgeLoss(WeightedLossFunctionWrapper):
    """Proposed in: 'CascadePSP: Toward Class-Agnostic and Very High-Resolution
    Segmentation via Global and Local Refinement'

    Implements Equation from https://arxiv.org/pdf/2005.02551
    Compute edge loss with Sobel operator
    """

    def __init__(
        self,
        from_logits=False,
        force_binary=False,
        reduction="sum_over_batch_size",
        name="sobel_edge_loss",
    ):
        super().__init__(
            sobel_edge_loss,
            reduction=reduction,
            name=name,
            from_logits=from_logits,
            force_binary=force_binary,
        )


def sobel(probs):
    classes = probs.shape[-1]
    sobel_x = np.reshape([[1, 0, -1], [2, 0, -2], [1, 0, -1]], [3, 3, 1, 1]) / 4
    sobel_x = np.tile(sobel_x, [1, 1, classes, 1])
    sobel_x = ops.cast(sobel_x, probs.dtype)

    sobel_y = np.reshape([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], [3, 3, 1, 1]) / 4
    sobel_y = np.tile(sobel_y, [1, 1, classes, 1])
    sobel_y = ops.cast(sobel_y, probs.dtype)

    pooled = ops.pad(probs, [(0, 0), (1, 1), (1, 1), (0, 0)], mode="SYMMETRIC")
    pooled = ops.average_pool(pooled, 3, strides=1, padding="valid")

    grad = ops.pad(pooled, [(0, 0), (1, 1), (1, 1), (0, 0)], mode="SYMMETRIC")
    grad_x = ops.depthwise_conv(grad, sobel_x, strides=1, padding="valid")
    grad_y = ops.depthwise_conv(grad, sobel_y, strides=1, padding="valid")

    edge = ops.sqrt(grad_x**2 + grad_y**2 + backend.epsilon())

    return edge


def sobel_edge_loss(y_true, y_pred, sample_weight, from_logits, force_binary):
    y_true, y_pred, sample_weight = validate_input(
        y_true, y_pred, sample_weight, dtype="int64", rank=4, channel="sparse"
    )
    y_pred, from_logits = to_probs(
        y_pred, from_logits, force_binary=force_binary
    )
    y_true, y_pred = to_1hot(y_true, y_pred, from_logits, dtype=y_pred.dtype)

    y_true_edge = sobel(y_true)
    y_true_edge = ops.stop_gradient(y_true_edge)
    y_pred_edge = sobel(y_pred)

    loss = ops.mean(ops.abs(y_true_edge - y_pred_edge), axis=-1, keepdims=True)
    loss = weighted_loss(loss, sample_weight)

    return loss
