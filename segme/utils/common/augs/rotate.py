import numpy as np
import tensorflow as tf
from segme.utils.common.augs.common import apply, transform, wrap, unwrap, validate
from segme.common.shape import get_shape


def rotate(image, masks, weight, prob, degrees, replace=None, name=None):
    with tf.name_scope(name or 'rotate'):
        return apply(
            image, masks, weight, prob,
            lambda x: _rotate(x, degrees, 'bilinear', replace),
            lambda x: _rotate(x, degrees, 'nearest', replace=np.zeros([1, 1, 1, x.shape[-1]])),
            lambda x: _rotate(x, degrees, 'nearest', replace=np.zeros([1, 1, 1, x.shape[-1]])))


def rotate_cw(image, masks, weight, prob, name=None):
    with tf.name_scope(name or 'rotate_cw'):
        return apply(
            image, masks, weight, prob,
            lambda x: _rotate_cw(x),
            lambda x: _rotate_cw(x),
            lambda x: _rotate_cw(x))


def rotate_ccw(image, masks, weight, prob, name=None):
    with tf.name_scope(name or 'rotate_ccw'):
        return apply(
            image, masks, weight, prob,
            lambda x: _rotate_ccw(x),
            lambda x: _rotate_ccw(x),
            lambda x: _rotate_ccw(x))


def _rotate(image, degrees, interpolation, replace=None, name=None):
    with tf.name_scope(name or 'rotate_'):
        image, _, _ = validate(image, None, None)

        radians = tf.cast(-degrees * np.pi / 180., 'float32')[None]
        (height, width), _ = get_shape(image, axis=[1, 2], dtype='float32')

        h_offset = ((width - 1) - (tf.cos(radians) * (width - 1) - tf.sin(radians) * (height - 1))) / 2.0
        v_offset = ((height - 1) - (tf.sin(radians) * (width - 1) + tf.cos(radians) * (height - 1))) / 2.0
        matrix = tf.concat(
            [tf.cos(radians)[:, None], -tf.sin(radians)[:, None], h_offset[:, None], tf.sin(radians)[:, None],
             tf.cos(radians)[:, None], v_offset[:, None], tf.zeros((1, 2), 'float32')],
            axis=1)

        image = wrap(image)
        image = transform(image, matrix, fill_mode='constant', interpolation=interpolation)
        image = unwrap(image, replace)

        return image


def _rotate_cw(image, name=None):
    with tf.name_scope(name or 'rotate_cw_'):
        image, _, _ = validate(image, None, None)

        image = tf.transpose(image, [0, 2, 1, 3])
        image = tf.reverse(image, [2])

        return image


def _rotate_ccw(image, name=None):
    with tf.name_scope(name or 'rotate_ccw_'):
        image, _, _ = validate(image, None, None)

        image = tf.transpose(image, [0, 2, 1, 3])
        image = tf.reverse(image, [1])

        return image
