import numpy as np
import tensorflow as tf
from keras_cv.utils import preprocessing
from segme.utils.common.augs.common import apply, validate, wrap, unwrap


def translatex(image, masks, weight, factor, prob, replace=None, name=None):
    with tf.name_scope(name or 'translatex'):
        return apply(
            image, masks, weight, prob,
            lambda x: _translatex(x, factor, 'bilinear', replace),
            lambda x: _translatex(x, factor, 'nearest', replace=np.zeros([1, 1, 1, x.shape[-1]])),
            lambda x: _translatex(x, factor, 'nearest', replace=np.zeros([1, 1, 1, x.shape[-1]])))


def translatey(image, masks, weight, factor, prob, replace=None, name=None):
    with tf.name_scope(name or 'translatey'):
        return apply(
            image, masks, weight, prob,
            lambda x: _translatey(x, factor, 'bilinear', replace),
            lambda x: _translatey(x, factor, 'nearest', replace=np.zeros([1, 1, 1, x.shape[-1]])),
            lambda x: _translatey(x, factor, 'nearest', replace=np.zeros([1, 1, 1, x.shape[-1]])))


def _translatex(image, factor, interpolation, replace=None, name=None):
    with tf.name_scope(name or 'translatex_'):
        image, _, _ = validate(image, None, None)

        width = tf.cast(tf.shape(image)[2], 'float32')
        translation = tf.stack([width * factor, 0])[None]
        transform = preprocessing.get_translation_matrix(translation)

        image = wrap(image)
        image = preprocessing.transform(image, transform, fill_mode='constant', interpolation=interpolation)
        image = unwrap(image, replace)

        return image


def _translatey(image, factor, interpolation, replace=None, name=None):
    with tf.name_scope(name or 'translatey_'):
        image, _, _ = validate(image, None, None)

        height = tf.cast(tf.shape(image)[1], 'float32')
        translation = tf.stack([0, height * factor])[None]
        transform = preprocessing.get_translation_matrix(translation)

        image = wrap(image)
        image = preprocessing.transform(image, transform, fill_mode='constant', interpolation=interpolation)
        image = unwrap(image, replace)

        return image
