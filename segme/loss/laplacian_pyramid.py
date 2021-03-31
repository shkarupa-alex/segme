import numpy as np
import tensorflow as tf
from .weighted_wrapper import WeightedLossFunctionWrapper


@tf.keras.utils.register_keras_serializable(package='SegMe')
class LaplacianPyramidLoss(WeightedLossFunctionWrapper):
    """ Proposed in: 'Optimizing the Latent Space of Generative Networks'

    Implements in https://arxiv.org/pdf/1707.05776.pdf
    """

    def __init__(
            self, levels=5, size=5, sigma=2.0, reduction=tf.keras.losses.Reduction.AUTO,
            name='laplacian_pyramid_loss'):
        super().__init__(
            laplacian_pyramid_loss, reduction=reduction, name=name, levels=levels, size=size, sigma=sigma)


def _gauss_kernel(size, sigma):
    grid = np.mgrid[0:size, 0:size].astype('float32').T
    gauss = np.exp((grid - size // 2) ** 2 / (-2 * sigma ** 2)) ** 2
    kernel = np.sum(gauss, axis=-1)
    kernel /= np.sum(kernel)

    return kernel


def _laplacian_pyramid(inputs, levels, kernel):
    pyramid = []

    outputs = inputs
    for level in range(levels):
        blurred = tf.nn.depthwise_conv2d(outputs, kernel, strides=[1, 1, 1, 1], padding='SAME')
        pyramid.append(outputs - blurred)
        outputs = tf.nn.avg_pool2d(blurred, 2, 2, 'VALID')
    pyramid.append(outputs)  # Low-frequency residual

    return pyramid


def laplacian_pyramid_loss(y_true, y_pred, sample_weight, levels, size, sigma):
    assert_true_rank = tf.assert_rank(y_true, 4)
    assert_pred_rank = tf.assert_rank(y_pred, 4)

    with tf.control_dependencies([assert_true_rank, assert_pred_rank]):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, dtype=y_pred.dtype)

        channels = int(y_pred.shape[-1])
        if channels is None:
            raise ValueError('Channel dimension of the predictions should be defined. Found `None`.')

        kernel = _gauss_kernel(size, sigma)[..., None, None]
        kernel = np.tile(kernel, (1, 1, channels, 1)).astype(y_pred.dtype.as_numpy_dtype)
        kernel = tf.constant(kernel, y_pred.dtype)

        pyr_true = _laplacian_pyramid(y_true, levels, kernel)
        pyr_pred = _laplacian_pyramid(y_pred, levels, kernel)

        if sample_weight is None:
            losses = [tf.abs(_true - _pred) for _true, _pred in zip(pyr_true, pyr_pred)]
        else:
            pyr_wght = _laplacian_pyramid(sample_weight, levels, kernel)
            losses = [tf.abs(_true - _pred) * _wght for _true, _pred, _wght in zip(pyr_true, pyr_pred, pyr_wght)]

        axis_hwc = list(range(1, y_pred.shape.ndims))
        losses = [(2 ** i / len(losses)) * tf.reduce_mean(l, axis=axis_hwc) for i, l in enumerate(losses)]
        losses = sum(losses)

        return losses
