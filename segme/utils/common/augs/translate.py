import numpy as np
import tensorflow as tf
from segme.utils.common.augs.common import apply, transform, validate, wrap, unwrap


def translate_x(image, masks, weight, prob, factor, replace=None, name=None):
    with tf.name_scope(name or 'translate_x'):
        return apply(
            image, masks, weight, prob,
            lambda x: _translate_x(x, factor, 'bilinear', replace),
            lambda x: _translate_x(x, factor, 'nearest', replace=np.zeros([1, 1, 1, x.shape[-1]])),
            lambda x: _translate_x(x, factor, 'nearest', replace=np.zeros([1, 1, 1, x.shape[-1]])))


def translate_y(image, masks, weight, prob, factor, replace=None, name=None):
    with tf.name_scope(name or 'translate_y'):
        return apply(
            image, masks, weight, prob,
            lambda x: _translate_y(x, factor, 'bilinear', replace),
            lambda x: _translate_y(x, factor, 'nearest', replace=np.zeros([1, 1, 1, x.shape[-1]])),
            lambda x: _translate_y(x, factor, 'nearest', replace=np.zeros([1, 1, 1, x.shape[-1]])))


def _translate_x(image, factor, interpolation, replace=None, name=None):
    with tf.name_scope(name or 'translate_x_'):
        image, _, _ = validate(image, None, None)

        width = tf.cast(tf.shape(image)[2], 'float32')
        translation = tf.stack([width * factor, 0])[None]
        matrix = tf.concat(
            [tf.ones((1, 1), 'float32'), tf.zeros((1, 1), 'float32'), -translation[:, 0, None],
             tf.zeros((1, 1), 'float32'), tf.ones((1, 1), 'float32'), -translation[:, 1, None],
             tf.zeros((1, 2), 'float32')],
            axis=1)

        image = wrap(image)
        image = transform(image, matrix, fill_mode='constant', interpolation=interpolation)
        image = unwrap(image, replace)

        return image


def _translate_y(image, factor, interpolation, replace=None, name=None):
    with tf.name_scope(name or 'translate_y_'):
        image, _, _ = validate(image, None, None)

        height = tf.cast(tf.shape(image)[1], 'float32')
        translation = tf.stack([0, height * factor])[None]
        matrix = tf.concat(
            [tf.ones((1, 1), 'float32'), tf.zeros((1, 1), 'float32'), -translation[:, 0, None],
             tf.zeros((1, 1), 'float32'), tf.ones((1, 1), 'float32'), -translation[:, 1, None],
             tf.zeros((1, 2), 'float32')],
            axis=1)

        image = wrap(image)
        image = transform(image, matrix, fill_mode='constant', interpolation=interpolation)
        image = unwrap(image, replace)

        return image
