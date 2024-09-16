import cv2
import numpy as np
from keras.src import ops
from keras.src.saving import register_keras_serializable

from segme.loss.common_loss import validate_input
from segme.loss.common_loss import weighted_loss
from segme.loss.weighted_wrapper import WeightedLossFunctionWrapper
from segme.ops import depth_to_space


@register_keras_serializable(package="SegMe>Loss")
class LaplacianPyramidLoss(WeightedLossFunctionWrapper):
    """Proposed in: 'Optimizing the Latent Space of Generative Networks'

    Implements Lap1 in https://arxiv.org/pdf/1707.05776
    """

    def __init__(
        self,
        levels=5,
        size=5,
        sigma=None,
        residual=False,
        weight_pooling="mean",
        reduction="sum_over_batch_size",
        name="laplacian_pyramid_loss",
    ):
        super().__init__(
            laplacian_pyramid_loss,
            reduction=reduction,
            name=name,
            levels=levels,
            size=size,
            sigma=sigma,
            residual=residual,
            weight_pooling=weight_pooling,
        )


def _pad_odd(inputs):
    height, width = ops.shape(inputs)[1:3]
    hpad, wpad = height % 2, width % 2
    paddings = [[0, 0], [0, hpad], [0, wpad], [0, 0]]
    padded = ops.pad(inputs, paddings, "SYMMETRIC")

    return padded


def _gauss_kernel(size, sigma):
    if sigma is None:
        sigma = 0.3 * ((size - 1) * 0.5 - 1) + 0.8
    kernel = cv2.getGaussianKernel(size, sigma)

    return kernel


def _gauss_filter(inputs, kernel):
    kernel_size = max(kernel[0].shape[0], kernel[1].shape[1])

    padding = kernel_size - 1
    padding = padding // 2, padding - padding // 2
    paddings = [(0, 0)] + [padding, padding] + [(0, 0)]

    padded = ops.pad(inputs, paddings, "REFLECT")
    blurred = ops.depthwise_conv(padded, kernel[0], strides=1, padding="valid")
    blurred = ops.depthwise_conv(blurred, kernel[1], strides=1, padding="valid")

    return blurred


def _gauss_downsample(inputs, kernel):
    blurred = _gauss_filter(inputs, kernel)
    downsampled = blurred[:, ::2, ::2, :]

    return downsampled


def _gauss_upsample(inputs, kernel):
    channels = inputs.shape[-1]
    if channels is None:
        raise ValueError(
            "Channel dimension of the inputs should be defined. Found `None`."
        )

    paddings = ((0, 0), (0, 0), (0, 0), (0, channels * 3))
    upsampled = ops.pad(inputs, paddings)
    upsampled = depth_to_space(upsampled, 2)

    return _gauss_filter(upsampled, (kernel[0] * 2.0, kernel[1] * 2.0))


def _laplacian_pyramid(inputs, levels, kernel, residual):
    # https://paperswithcode.com/method/laplacian-pyramid
    pyramid = []

    current = inputs
    for level in range(levels):
        current = _pad_odd(current)
        downsampled = _gauss_downsample(current, kernel)
        upsampled = _gauss_upsample(downsampled, kernel)
        pyramid.append(current - upsampled)
        current = downsampled

    if residual:
        pyramid.append(current)

    return pyramid


def _weight_pyramid(inputs, levels, residual, weight_pooling):
    if inputs is None:
        return [None] * (levels + int(residual))

    if weight_pooling in {"min", "max"}:
        pooling = ops.max_pool
    elif "mean" == weight_pooling:
        pooling = ops.average_pool
    else:
        raise ValueError("Unknown weight pooling mode")

    pyramid = []
    current = inputs
    for level in range(levels):
        current = _pad_odd(current)
        pyramid.append(ops.stop_gradient(current))

        if "min" == weight_pooling:
            current = -current
        current = pooling(current, 2, strides=2, padding="valid")
        if "min" == weight_pooling:
            current = -current

    if residual:
        pyramid.append(ops.stop_gradient(current))

    return pyramid


def laplacian_pyramid_loss(
    y_true, y_pred, sample_weight, levels, size, sigma, residual, weight_pooling
):
    y_true, y_pred, sample_weight = validate_input(
        y_true, y_pred, sample_weight, dtype=None, rank=4, channel="same"
    )

    kernel = _gauss_kernel(size, sigma)
    kernel = np.tile(kernel[..., None, None], (1, 1, y_pred.shape[-1], 1))
    kernel = ops.cast(kernel, y_pred.dtype), ops.cast(
        kernel.transpose([1, 0, 2, 3]), y_pred.dtype
    )

    height, width = ops.shape(y_true)[1:3]
    if isinstance(height, int) and isinstance(width, int):
        if height <= 2**levels or width <= 2**levels:
            raise ValueError(
                "Laplacian pyramid loss does not support "
                "inputs with spatial size <= 2^levels."
            )

    pyr_pred = _laplacian_pyramid(y_pred, levels, kernel, residual)
    pyr_true = _laplacian_pyramid(y_true, levels, kernel, residual)
    pyr_true = [ops.stop_gradient(pt) for pt in pyr_true]

    losses = [
        ops.abs(_true - _pred) for _true, _pred in zip(pyr_true, pyr_pred)
    ]
    weights = _weight_pyramid(sample_weight, levels, residual, weight_pooling)
    losses = [
        weighted_loss(loss, weight) for loss, weight in zip(losses, weights)
    ]

    losses = [loss * (2**i) for i, loss in enumerate(losses)]
    losses = sum(losses)

    return losses
