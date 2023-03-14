import numpy as np
import tensorflow as tf
from keras_cv.utils import preprocessing
from segme.utils.common.augs.common import apply, validate, wrap, unwrap


def shearx(image, masks, weight, factor, prob, replace=None, name=None):
    with tf.name_scope(name or 'shearx'):
        return apply(
            image, masks, weight, prob,
            lambda x: _shearx(x, factor, 'bilinear', replace),
            lambda x: _shearx(x, factor, 'nearest', replace=np.zeros([1, 1, 1, x.shape[-1]])),
            lambda x: _shearx(x, factor, 'nearest', replace=np.zeros([1, 1, 1, x.shape[-1]])))


def sheary(image, masks, weight, factor, prob, replace=None, name=None):
    with tf.name_scope(name or 'sheary'):
        return apply(
            image, masks, weight, prob,
            lambda x: _sheary(x, factor, 'bilinear', replace),
            lambda x: _sheary(x, factor, 'nearest', replace=np.zeros([1, 1, 1, x.shape[-1]])),
            lambda x: _sheary(x, factor, 'nearest', replace=np.zeros([1, 1, 1, x.shape[-1]])))


def _shearx(image, factor, interpolation, replace=None, name=None):
    with tf.name_scope(name or 'shearx_'):
        image, _, _ = validate(image, None, None)

        transform = tf.stack([1., factor, 0., 0., 1., 0., 0., 0.])[None]

        image = wrap(image)
        image = preprocessing.transform(image, transform, fill_mode='constant', interpolation=interpolation)
        image = unwrap(image, replace)

        return image


def _sheary(image, factor, interpolation, replace=None, name=None):
    with tf.name_scope(name or 'sheary_'):
        image, _, _ = validate(image, None, None)

        transform = tf.stack([1., 0., 0., factor, 1., 0., 0., 0.])[None]

        image = wrap(image)
        image = preprocessing.transform(image, transform, fill_mode='constant', interpolation=interpolation)
        image = unwrap(image, replace)

        return image
