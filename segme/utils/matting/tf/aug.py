import tensorflow as tf
from segme.utils.common.augs.common import convert
from segme.common.shape import get_shape


def augment_alpha(alpha, prob=0.3, max_pow=2., seed=None):
    with tf.name_scope('augment_alpha'):
        alpha = tf.convert_to_tensor(alpha, 'uint8')

        if 4 != alpha.shape.rank:
            raise ValueError('Expecting `alpha` rank to be 4.')

        if 1 != alpha.shape[-1]:
            raise ValueError('Expecting `alpha` channels size to be 1.')

        if 'uint8' != alpha.dtype:
            raise ValueError('Expecting `alpha` dtype to be `uint8`.')

        alpha = convert(alpha, 'float32')

        (batch,), _ = get_shape(alpha, axis=[0])
        gamma, direction, apply, invert = tf.split(
            tf.random.uniform([batch, 1, 1, 4], 0., 1., seed=seed), 4, axis=-1)

        direction = tf.cast(direction > 0.5, direction.dtype)
        gamma = gamma * (max_pow - 1.) + 1.
        gamma = direction * gamma + (1. - direction) / gamma

        invert = tf.cast(invert > 0.5, invert.dtype)
        alpha_ = (1. - alpha) * invert + alpha * (1. - invert)
        alpha_ = tf.pow(alpha_, gamma)
        alpha_ = (1. - alpha_) * invert + alpha_ * (1. - invert)

        apply = tf.cast(apply < prob, alpha.dtype)
        alpha = alpha_ * apply + alpha * (1 - apply)

        return convert(alpha, 'uint8', saturate=True)


def augment_trimap(trimap, prob=0.1, seed=None):
    with tf.name_scope('augment_trimap'):
        trimap = tf.convert_to_tensor(trimap, 'uint8')

        if 4 != trimap.shape.rank:
            raise ValueError('Expecting `trimap` rank to be 4.')

        if 1 != trimap.shape[-1]:
            raise ValueError('Expecting `trimap` channels size to be 1.')

        if 'uint8' != trimap.dtype:
            raise ValueError('Expecting `trimap` dtype to be `uint8`.')

        (batch,), _ = get_shape(trimap, axis=[0])
        apply = tf.random.uniform([batch, 1, 1, 1], 0., 1., seed=seed)
        apply = tf.cast(apply < prob, trimap.dtype)

        trimap_ = tf.minimum(trimap, 128)

        return trimap_ * apply + trimap * (1 - apply)
