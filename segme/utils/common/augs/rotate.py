import numpy as np
import tensorflow as tf
from keras_cv.utils import preprocessing
from segme.utils.common.augs.common import apply, wrap, unwrap, validate


def rotate(image, masks, weight, degrees, prob, replace=None, name=None):
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
        height, width = tf.unstack(tf.cast(tf.shape(image)[1:3], 'float32'))
        transform = preprocessing.get_rotation_matrix(radians, height, width)

        image = wrap(image)
        image = preprocessing.transform(image, transform, fill_mode='constant', interpolation=interpolation)
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
