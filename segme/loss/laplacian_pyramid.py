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

    def __init__(
            self, levels=5, size=5, sigma=2.0, reduction=Reduction.AUTO,
            name='laplacian_pyramid_loss'):
        super().__init__(
            laplacian_pyramid_loss, reduction=reduction, name=name, levels=levels, size=size, sigma=sigma)


def _gauss_kernel(size, sigma):
    # Implements [9] in 'Diffusion Distance for Histogram Comparison', OI 10.1109/CVPR.2006.99
    space = np.arange(size) - (size - 1) / 2

    sigma2 = sigma ** 2
    gauss1d = np.exp(-space ** 2 / (2 * sigma2)) / np.sqrt(2 * np.pi * sigma2)

    gauss2d = gauss1d[..., None] * gauss1d[None, ...]

    gauss2d /= np.sum(gauss2d)

    return gauss2d


def _gauss_filter(inputs, kernel):
    paddings = [((k - 1) // 2, k // 2) for k in kernel.shape[:2][::-1]]
    paddings = [(0, 0)] + paddings + [(0, 0)]

    padded = tf.pad(inputs, paddings, 'REFLECT')
    blurred = tf.nn.depthwise_conv2d(padded, kernel, strides=[1, 1, 1, 1], padding='VALID')

    return blurred


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

    return _gauss_filter(upsampled, kernel * 4.)


def _laplacian_pyramid(inputs, levels, kernel):
    # https://paperswithcode.com/method/laplacian-pyramid
    pyramid = []

    current = inputs
    for level in range(levels):
        downsampled = _gauss_downsample(current, kernel)
        upsampled = _gauss_upsample(downsampled, kernel)
        pyramid.append(current - upsampled)
        current = downsampled
    pyramid.append(current)  # Low-frequency residual

    return pyramid


def laplacian_pyramid_loss(y_true, y_pred, sample_weight, levels, size, sigma):
    assert_true_rank = tf.assert_rank(y_true, 4)
    assert_pred_rank = tf.assert_rank(y_pred, 4)

    with tf.control_dependencies([assert_true_rank, assert_pred_rank]):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, dtype=y_pred.dtype)

        channels_pred = y_pred.shape[-1]
        if channels_pred is None:
            raise ValueError('Channel dimension of the predictions should be defined. Found `None`.')

        kernel = _gauss_kernel(size, sigma)[..., None, None]
        kernel = kernel.astype(y_pred.dtype.as_numpy_dtype)

        kernel_pred = np.tile(kernel, (1, 1, channels_pred, 1))
        kernel_pred = tf.constant(kernel_pred, y_pred.dtype)
        pyr_pred = _laplacian_pyramid(y_pred, levels, kernel_pred)
        pyr_true = _laplacian_pyramid(y_true, levels, kernel_pred)
        pyr_true = [tf.stop_gradient(pt) for pt in pyr_true]

        if sample_weight is None:
            losses = [tf.abs(_true - _pred) for _true, _pred in zip(pyr_true, pyr_pred)]
        else:
            channels_wght = sample_weight.shape[-1]
            if channels_wght is None:
                raise ValueError('Channel dimension of the sample weights should be defined. Found `None`.')

            kernel_wght = np.tile(kernel, (1, 1, channels_wght, 1))
            kernel_wght = tf.constant(kernel_wght, y_pred.dtype)
            pyr_wght = _laplacian_pyramid(sample_weight, levels, kernel_wght)
            losses = [tf.abs(_true - _pred) * _wght for _true, _pred, _wght in zip(pyr_true, pyr_pred, pyr_wght)]

        axis_hwc = list(range(1, y_pred.shape.ndims))
        losses = [(2 ** i / len(losses)) * tf.reduce_mean(l, axis=axis_hwc) for i, l in enumerate(losses)]
        losses = sum(losses)

        return losses
