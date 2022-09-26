import cv2
import numpy as np
import tensorflow as tf
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.losses_utils import ReductionV2 as Reduction
from segme.loss.common_loss import validate_input, weighted_loss
from segme.loss.weighted_wrapper import WeightedLossFunctionWrapper


@register_keras_serializable(package='SegMe>Loss')
class LaplacianPyramidLoss(WeightedLossFunctionWrapper):
    """ Proposed in: 'Optimizing the Latent Space of Generative Networks'

    Implements Lap1 in https://arxiv.org/pdf/1707.05776.pdf
    """

    def __init__(self, levels=5, size=5, sigma=None, residual=False, weight_pooling='mean', reduction=Reduction.AUTO,
                 name='laplacian_pyramid_loss'):
        super().__init__(laplacian_pyramid_loss, reduction=reduction, name=name, levels=levels, size=size, sigma=sigma,
                         residual=residual, weight_pooling=weight_pooling)


def _pad_odd(inputs):
    height_width = tf.shape(inputs)[1:3]
    hpad, wpad = tf.unstack(height_width % 2)
    paddings = [[0, 0], [0, hpad], [0, wpad], [0, 0]]
    padded = tf.pad(inputs, paddings, 'SYMMETRIC')

    return padded


def _gauss_kernel(size, sigma):
    if sigma is None:
        sigma = 0.3 * ((size - 1) * 0.5 - 1) + 0.8
    kernel = cv2.getGaussianKernel(size, sigma)

    return kernel


def _gauss_filter(inputs, kernel):
    kernel_size = max(kernel[0].shape[0], kernel[1].shape[1])

    padding = (kernel_size - 1)
    padding = padding // 2, padding - padding // 2
    paddings = [(0, 0)] + [padding, padding] + [(0, 0)]

    padded = tf.pad(inputs, paddings, 'REFLECT')
    blurred = tf.nn.depthwise_conv2d(padded, kernel[0], strides=[1, 1, 1, 1], padding='VALID')
    blurred = tf.nn.depthwise_conv2d(blurred, kernel[1], strides=[1, 1, 1, 1], padding='VALID')

    return blurred


def _gauss_downsample(inputs, kernel):
    blurred = _gauss_filter(inputs, kernel)
    downsampled = blurred[:, ::2, ::2, :]

    return downsampled


def _gauss_upsample(inputs, kernel):
    channels = inputs.shape[-1]
    if channels is None:
        raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')

    paddings = ((0, 0), (0, 0), (0, 0), (0, channels * 3))
    upsampled = tf.pad(inputs, paddings)
    upsampled = tf.nn.depth_to_space(upsampled, 2)

    return _gauss_filter(upsampled, (kernel[0] * 2., kernel[1] * 2.))


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

    if weight_pooling in {'min', 'max'}:
        pooling = tf.nn.max_pool2d
    elif 'mean' == weight_pooling:
        pooling = tf.nn.avg_pool2d
    else:
        raise ValueError('Unknown weight pooling mode')

    pyramid = []
    current = inputs
    for level in range(levels):
        current = _pad_odd(current)
        pyramid.append(current)

        if 'min' == weight_pooling:
            current = -current
        current = pooling(current, ksize=2, strides=2, padding='VALID')
        if 'min' == weight_pooling:
            current = -current

    if residual:
        pyramid.append(current)

    return pyramid


def laplacian_pyramid_loss(y_true, y_pred, sample_weight, levels, size, sigma, residual, weight_pooling):
    y_true, y_pred, sample_weight = validate_input(
        y_true, y_pred, sample_weight, dtype=None, rank=4, channel='same')

    kernel = _gauss_kernel(size, sigma)
    kernel = np.tile(kernel[..., None, None], (1, 1, y_pred.shape[-1], 1))
    kernel = tf.cast(kernel, y_pred.dtype), tf.cast(kernel.transpose([1, 0, 2, 3]), y_pred.dtype)

    assert_true_shape = tf.assert_greater(tf.reduce_min(tf.shape(y_true)[1:3]), 2 ** levels)
    with tf.control_dependencies([assert_true_shape]):
        pyr_pred = _laplacian_pyramid(y_pred, levels, kernel, residual)
        pyr_true = _laplacian_pyramid(y_true, levels, kernel, residual)
        pyr_true = [tf.stop_gradient(pt) for pt in pyr_true]

    losses = [tf.abs(_true - _pred) for _true, _pred in zip(pyr_true, pyr_pred)]
    weights = _weight_pyramid(sample_weight, levels, residual, weight_pooling)
    losses = [weighted_loss(loss, weight) for loss, weight in zip(losses, weights)]

    losses = [loss * (2 ** i) for i, loss in enumerate(losses)]
    losses = sum(losses)

    return losses
