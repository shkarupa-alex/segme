import numpy as np
import tensorflow as tf
from keras import backend
from scipy.special import softmax as np_softmax
from keras.saving.object_registration import register_keras_serializable
from keras.utils.losses_utils import ReductionV2 as Reduction
from tensorflow.python.ops.image_ops_impl import _ssim_helper
from segme.loss.common_loss import validate_input
from segme.loss.weighted_wrapper import WeightedLossFunctionWrapper


@register_keras_serializable(package='SegMe>Loss')
class StructuralSimilarityLoss(WeightedLossFunctionWrapper):
    """ Proposed in: 'Optimizing the Latent Space of Generative Networks'

    Implements Lap1 in https://arxiv.org/pdf/1707.05776.pdf
    """

    def __init__(self, max_val=1., factors=(0.0448, 0.2856, 0.3001, 0.2363, 0.1333), size=11, sigma=2.0, k1=0.01,
                 k2=0.03, weight_pooling='mean', reduction=Reduction.AUTO, name='structural_similarity_loss'):
        super().__init__(structural_similarity_loss, reduction=reduction, name=name, max_val=max_val, factors=factors,
                         size=size, sigma=sigma, k1=k1, k2=k2, weight_pooling=weight_pooling)


def _ssim_level(y_true, y_pred, max_val, kernels, k1, k2):
    # The correct compensation factor is `1.0 - tf.reduce_sum(tf.square(kernel))`,
    # but MATLAB implementation of MS-SSIM uses just 1.0
    compensation = 1.0 - tf.reduce_sum(tf.square(kernels[0] * kernels[1]))

    def _reducer(x):
        x = tf.nn.depthwise_conv2d(x, kernels[0], strides=[1, 1, 1, 1], padding='VALID')
        x = tf.nn.depthwise_conv2d(x, kernels[1], strides=[1, 1, 1, 1], padding='VALID')

        return x

    luminance, contrast_structure = _ssim_helper(
        y_true, y_pred, _reducer, max_val, compensation, k1, k2)

    similarity = luminance * contrast_structure

    return similarity, contrast_structure


def _pad_odd(inputs):
    height_width = tf.shape(inputs)[1:3]
    hpad, wpad = tf.unstack(height_width % 2)
    paddings = [[0, 0], [0, hpad], [0, wpad], [0, 0]]
    padded = tf.pad(inputs, paddings, 'SYMMETRIC')

    return padded


def _ssim_pyramid(y_true, y_pred, sample_weight, max_val, factors, kernels, k1, k2, weight_pooling):
    ksize = (kernels[0].shape[0], kernels[1].shape[1])

    if weight_pooling in {'min', 'max'}:
        pooling = tf.nn.max_pool2d
    elif 'mean' == weight_pooling:
        pooling = tf.nn.avg_pool2d
    else:
        raise ValueError('Unknown weight pooling mode')

    pyramid = []
    for i, f in enumerate(factors):
        last_level = len(factors) - 1 == i

        similarity, contrast_structure = _ssim_level(
            y_true, y_pred, max_val=max_val, kernels=kernels, k1=k1, k2=k2)
        value = similarity if last_level else contrast_structure
        value = tf.nn.relu(value) ** f

        if sample_weight is not None:
            if 'min' == weight_pooling:
                sample_weight = -sample_weight
            weight = pooling(sample_weight, ksize=ksize, strides=1, padding='VALID')
            if 'min' == weight_pooling:
                weight = -weight

            weight = tf.cast(weight > 0., y_pred.dtype)
            weight = tf.stop_gradient(weight)

            value *= weight
            value = tf.math.divide_no_nan(
                tf.reduce_sum(value, axis=[1, 2]),
                tf.reduce_sum(weight, axis=[1, 2]))
        else:
            value = tf.reduce_mean(value, axis=[1, 2])
        pyramid.append(value)

        if not last_level:
            assert_true_shape = tf.assert_greater(tf.reduce_min(tf.shape(y_true)[1:3]), 2)
            with tf.control_dependencies([assert_true_shape]):
                y_true = _pad_odd(y_true)
                y_true = tf.nn.avg_pool2d(y_true, ksize=2, strides=2, padding='VALID')

                y_pred = _pad_odd(y_pred)
                y_pred = tf.nn.avg_pool2d(y_pred, ksize=2, strides=2, padding='VALID')

                if sample_weight is not None:
                    sample_weight = _pad_odd(sample_weight)
                    if 'min' == weight_pooling:
                        sample_weight = -sample_weight
                    sample_weight = pooling(sample_weight, ksize=2, strides=2, padding='VALID')
                    if 'min' == weight_pooling:
                        sample_weight = -sample_weight

    pyramid = tf.stack(pyramid, axis=-1)
    pyramid = tf.reduce_prod(pyramid, [-1])

    return tf.reduce_mean(pyramid, [-1])


def _separated_fspecial_gauss(size, sigma):
    coords = np.arange(size, dtype='float32')
    coords -= (size - 1) / 2.

    g = coords ** 2
    g *= -0.5 / (sigma ** 2)

    g = g.reshape((1, -1)) + g.reshape((-1, 1))
    g = g.reshape((1, -1))  # For tf.nn.softmax().
    g = np_softmax(g, axis=-1)

    g = g.reshape((size, size))
    u, e, v = np.linalg.svd(g)
    assert 1 == np.sum(e > 1e-5)

    e0 = np.sqrt(e[0])
    v0 = u[:, :1] * e0
    h0 = v[:1, :] * e0
    assert np.abs(g - h0 * v0).max() < 1e-5, np.abs(g - h0 * v0).max()

    return v0.reshape((size, 1, 1, 1)), h0.reshape((1, size, 1, 1))


def _ssim_kernel(size, sigma, channels, dtype):
    kernel0, kernel1 = _separated_fspecial_gauss(size, sigma)
    kernel0 = tf.tile(tf.cast(kernel0, dtype), multiples=[1, 1, channels, 1])
    kernel1 = tf.tile(tf.cast(kernel1, dtype), multiples=[1, 1, channels, 1])

    return kernel0, kernel1


def structural_similarity_loss(y_true, y_pred, sample_weight, max_val, factors, size, sigma, k1, k2, weight_pooling):
    y_true, y_pred, sample_weight = validate_input(
        y_true, y_pred, sample_weight, dtype=None, rank=4, channel='same')

    kernels = _ssim_kernel(size, sigma, y_pred.shape[-1], y_pred.dtype)

    max_delta = tf.reduce_max(y_true) - tf.reduce_min(y_true)
    assert_true_delta = tf.assert_less(max_delta, max_val + backend.epsilon())
    with tf.control_dependencies([assert_true_delta]):
        loss = 1. - _ssim_pyramid(y_true, y_pred, sample_weight, max_val, factors, kernels, k1, k2, weight_pooling)

    if sample_weight is not None:
        valid_weight = tf.cast(sample_weight > 0., y_pred.dtype)
        batch_weight = tf.math.divide_no_nan(
            tf.reduce_sum(sample_weight, axis=[1, 2, 3]),
            tf.reduce_sum(valid_weight, axis=[1, 2, 3]) + backend.epsilon())
        loss *= batch_weight

    return loss
