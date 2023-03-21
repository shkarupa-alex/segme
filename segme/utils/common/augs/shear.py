import numpy as np
import tensorflow as tf
from segme.utils.common.augs.common import apply, transform, validate, wrap, unwrap


def shear_x(image, masks, weight, prob, factor, replace=None, name=None):
    with tf.name_scope(name or 'shear_x'):
        return apply(
            image, masks, weight, prob,
            lambda x: _shear_x(x, factor, 'bilinear', replace),
            lambda x: _shear_x(x, factor, 'nearest', replace=np.zeros([1, 1, 1, x.shape[-1]])),
            lambda x: _shear_x(x, factor, 'nearest', replace=np.zeros([1, 1, 1, x.shape[-1]])))


def shear_y(image, masks, weight, prob, factor, replace=None, name=None):
    with tf.name_scope(name or 'shear_y'):
        return apply(
            image, masks, weight, prob,
            lambda x: _shear_y(x, factor, 'bilinear', replace),
            lambda x: _shear_y(x, factor, 'nearest', replace=np.zeros([1, 1, 1, x.shape[-1]])),
            lambda x: _shear_y(x, factor, 'nearest', replace=np.zeros([1, 1, 1, x.shape[-1]])))


def _shear_x(image, factor, interpolation, replace=None, name=None):
    with tf.name_scope(name or 'shear_x_'):
        image, _, _ = validate(image, None, None)

        matrix = tf.stack([1., factor, 0., 0., 1., 0., 0., 0.])[None]

        image = wrap(image)
        image = transform(image, matrix, fill_mode='constant', interpolation=interpolation)
        image = unwrap(image, replace)

        return image


def _shear_y(image, factor, interpolation, replace=None, name=None):
    with tf.name_scope(name or 'shear_y_'):
        image, _, _ = validate(image, None, None)

        matrix = tf.stack([1., 0., 0., factor, 1., 0., 0., 0.])[None]

        image = wrap(image)
        image = transform(image, matrix, fill_mode='constant', interpolation=interpolation)
        image = unwrap(image, replace)

        return image
