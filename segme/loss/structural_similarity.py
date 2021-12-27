import tensorflow as tf
from keras import backend
from tensorflow.python.ops.image_ops_impl import _fspecial_gauss, _ssim_helper
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.losses_utils import ReductionV2 as Reduction
from .weighted_wrapper import WeightedLossFunctionWrapper


@register_keras_serializable(package='SegMe')
class StructuralSimilarityLoss(WeightedLossFunctionWrapper):
    """ Proposed in: 'Optimizing the Latent Space of Generative Networks'

    Implements Lap1 in https://arxiv.org/pdf/1707.05776.pdf
    """

    def __init__(
            self, max_val=1., factors=(0.0448, 0.2856, 0.3001, 0.2363, 0.1333), size=11, sigma=2.0, k1=0.01, k2=0.03,
            reduction=Reduction.AUTO, name='structural_similarity_loss'):
        super().__init__(
            structural_similarity_loss, reduction=reduction, name=name, max_val=max_val, factors=factors, size=size,
            sigma=sigma, k1=k1, k2=k2)


def _ssim_level(y_true, y_pred, max_val, kernel, k1, k2):
    # The correct compensation factor is `1.0 - tf.reduce_sum(tf.square(kernel))`,
    # but MATLAB implementation of MS-SSIM uses just 1.0
    compensation = 1.0 - tf.reduce_sum(tf.square(kernel))

    luminance, contrast_structure = _ssim_helper(
        y_true, y_pred, lambda x: tf.nn.depthwise_conv2d(x, kernel, strides=[1, 1, 1, 1], padding='VALID'),
        max_val, compensation, k1, k2)

    similarity = luminance * contrast_structure

    return similarity, contrast_structure


def _pad_odd(inputs):
    height_width = tf.shape(inputs)[1:3]
    hpad, wpad = tf.unstack(height_width % 2)
    paddings = [[0, 0], [0, hpad], [0, wpad], [0, 0]]
    padded = tf.pad(inputs, paddings, 'REFLECT')

    return padded


def _ssim_pyramid(y_true, y_pred, sample_weight, max_val, factors, kernel, k1, k2):
    pyramid = []
    for i, f in enumerate(factors):
        similarity, contrast_structure = _ssim_level(
            y_true, y_pred, max_val=max_val, kernel=kernel, k1=k1, k2=k2)
        if sample_weight is not None:
            sample_weight = tf.nn.avg_pool(sample_weight, ksize=kernel.shape[:2], strides=1, padding='VALID')
        last_level = len(factors) - 1 == i

        value = similarity if last_level else contrast_structure
        value = value if sample_weight is None else value * sample_weight
        value = tf.reduce_mean(value, axis=[1, 2])
        if tf.executing_eagerly():
            print(value.numpy())
        value = tf.nn.relu(value) ** f
        pyramid.append(value)

        if not last_level:
            y_true = _pad_odd(y_true)
            y_true = tf.nn.avg_pool(y_true, ksize=2, strides=2, padding='VALID')

            y_pred = _pad_odd(y_pred)
            y_pred = tf.nn.avg_pool(y_pred, ksize=2, strides=2, padding='VALID')

            if sample_weight is not None:
                sample_weight = _pad_odd(sample_weight)
                sample_weight = tf.nn.avg_pool(sample_weight, ksize=2, strides=2, padding='VALID')

    pyramid = tf.stack(pyramid, axis=-1)
    pyramid = tf.reduce_prod(pyramid, [-1])

    return tf.reduce_mean(pyramid, [-1])


def _ssim_kernel(size, sigma, channels, dtype):
    kernel = _fspecial_gauss(size, sigma)
    kernel = tf.tile(kernel, multiples=[1, 1, channels, 1])
    kernel = tf.cast(kernel, dtype)
    kernel = tf.constant(kernel)

    return kernel


def structural_similarity_loss(y_true, y_pred, sample_weight, max_val, factors, size, sigma, k1, k2):
    assert_true_rank = tf.assert_rank(y_true, 4)
    assert_pred_rank = tf.assert_rank(y_pred, 4)
    assert_true_shape = tf.assert_greater(tf.reduce_min(tf.shape(y_true)[1:3]), size * 2 ** (len(factors) - 1))
    assert_true_delta = tf.assert_less(tf.reduce_max(y_true) - tf.reduce_min(y_true), max_val + backend.epsilon())

    with tf.control_dependencies([assert_true_rank, assert_pred_rank, assert_true_shape, assert_true_delta]):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, dtype=y_pred.dtype)

        channels_pred = y_pred.shape[-1]
        if channels_pred is None:
            raise ValueError('Channel dimension of the predictions should be defined. Found `None`.')

        kernel = _ssim_kernel(size, sigma, channels_pred, y_pred.dtype)
        losses = 1. - _ssim_pyramid(y_true, y_pred, sample_weight, max_val, factors, kernel, k1, k2)

        return losses
