import cv2
import numpy as np
import tensorflow as tf
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.losses_utils import ReductionV2 as Reduction
from .weighted_wrapper import WeightedLossFunctionWrapper


@register_keras_serializable(package='SegMe')
class LaplacianPyramidLoss(WeightedLossFunctionWrapper):
    """ Proposed in: 'Optimizing the Latent Space of Generative Networks'

    Implements Lap1 in https://arxiv.org/pdf/1707.05776.pdf
    """

    def __init__(self, levels=5, size=5, sigma=None, reduction=Reduction.AUTO, name='laplacian_pyramid_loss'):
        super().__init__(laplacian_pyramid_loss, reduction=reduction, name=name, levels=levels, size=size, sigma=sigma)


def _pad_odd(inputs):
    height_width = tf.shape(inputs)[1:3]
    hpad, wpad = tf.unstack(height_width % 2)
    paddings = [[0, 0], [0, hpad], [0, wpad], [0, 0]]
    padded = tf.pad(inputs, paddings, 'REFLECT')

    return padded


def _gauss_kernel(size, sigma):
    if sigma is None:
        sigma = 0.3 * ((size - 1) * 0.5 - 1) + 0.8
    kernel = cv2.getGaussianKernel(size, sigma)

    return kernel


def _gauss_filter(inputs, kernel):
    kernel_shape = (kernel[0].shape[0], kernel[1].shape[1])
    paddings = [((k - 1) // 2, k // 2) for k in kernel_shape[::-1]]
    paddings = [(0, 0)] + paddings + [(0, 0)]

    padded = tf.pad(inputs, paddings, 'REFLECT')
    blurred = tf.nn.depthwise_conv2d(padded, kernel[0], strides=[1, 1, 1, 1], padding='VALID')
    blurred = tf.nn.depthwise_conv2d(blurred, kernel[1], strides=[1, 1, 1, 1], padding='VALID')

    return blurred


def _regular_downsample(inputs):
    size = tf.shape(inputs)[1:3] // 2
    down = tf.image.resize(inputs, size)

    return down


def _gauss_downsample(inputs, kernel):
    blurred = _gauss_filter(inputs, kernel)
    downsampled = blurred[:, ::2, ::2, :]

    return downsampled


def _gauss_upsample(inputs, kernel):
    shape = tf.shape(inputs)
    batch, height, width, channel = tf.unstack(shape)

    upsampled = tf.concat([inputs, tf.zeros_like(inputs)], axis=-1)
    upsampled = tf.reshape(upsampled, [batch * height, width * 2 * channel])

    upsampled = tf.concat([upsampled, tf.zeros_like(upsampled)], axis=-1)
    upsampled = tf.reshape(upsampled, [batch, height * 2, width * 2, channel])

    return _gauss_filter(upsampled, (kernel[0] * 2., kernel[1] * 2.))


def _laplacian_pyramid(inputs, levels, kernel):
    # https://paperswithcode.com/method/laplacian-pyramid
    pyramid = []

    current = inputs
    for level in range(levels):
        current = _pad_odd(current)
        downsampled = _gauss_downsample(current, kernel)
        upsampled = _gauss_upsample(downsampled, kernel)
        pyramid.append(current - upsampled)
        current = downsampled

    # Disabled: low-frequency residual
    # pyramid.append(current)

    return pyramid


def _weight_pyramid(inputs, levels):
    # https://gist.github.com/MarcoForte/a07c40a2b721739bb5c5987671aa5270
    pyramid = []

    current = inputs
    for level in range(levels):
        current = _pad_odd(current)
        pyramid.append(current)
        current = _regular_downsample(current)

    # Disabled: low-frequency residual
    # pyramid.append(current)

    return pyramid


def laplacian_pyramid_loss(y_true, y_pred, sample_weight, levels, size, sigma):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, dtype=y_pred.dtype)

    channels_pred = y_pred.shape[-1]
    if channels_pred is None:
        raise ValueError('Channel dimension of the predictions should be defined. Found `None`.')

    assert_true_rank = tf.assert_rank(y_true, 4)
    assert_pred_rank = tf.assert_rank(y_pred, 4)

    with tf.control_dependencies([assert_true_rank, assert_pred_rank]):
        kernel = _gauss_kernel(size, sigma)
        kernel = np.tile(kernel[..., None, None], (1, 1, channels_pred, 1))
        kernel = tf.cast(kernel, y_pred.dtype), tf.cast(kernel.transpose([1, 0, 2, 3]), y_pred.dtype)

        pyr_pred = _laplacian_pyramid(y_pred, levels, kernel)
        pyr_true = _laplacian_pyramid(y_true, levels, kernel)
        pyr_true = [tf.stop_gradient(pt) for pt in pyr_true]

        losses = [tf.abs(_true - _pred) for _true, _pred in zip(pyr_true, pyr_pred)]
        if sample_weight is not None:
            weights = _weight_pyramid(sample_weight, levels)
            weights = [tf.stop_gradient(pw) for pw in weights]
            losses = [ls * sw for ls, sw in zip(losses, weights)]

        axis_hwc = list(range(1, y_pred.shape.ndims))
        losses = [tf.reduce_mean(l, axis=axis_hwc) * (2 ** i) for i, l in enumerate(losses)]
        losses = sum(losses)

        return losses
