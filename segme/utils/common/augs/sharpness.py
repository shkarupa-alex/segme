import numpy as np
import tensorflow as tf
from segme.utils.common.augs.common import apply, blend, convert, validate


def sharpness(image, masks, weight, prob, factor, name=None):
    with tf.name_scope(name or 'sharpness'):
        return apply(
            image, masks, weight, prob,
            lambda x: _sharpness(x, factor),
            lambda x: tf.identity(x),
            lambda x: tf.identity(x))


def _sharpness(image, factor, name=None):
    with tf.name_scope(name or 'sharpness_'):
        image, _, _ = validate(image, None, None)

        dtype = image.dtype
        image = convert(image, 'float32')

        kernel = np.array([[1, 1, 1], [1, 5, 1], [1, 1, 1]], 'float32') / 13.
        kernel = np.tile(kernel[..., None, None], [1, 1, 3, 1])
        kernel = tf.cast(kernel, 'float32')

        image_ = tf.nn.depthwise_conv2d(image, kernel, [1] * 4, padding='VALID')
        image_ = tf.clip_by_value(image_, 0., 1.)

        mask = tf.ones_like(image_)
        mask = tf.pad(mask, [[0, 0], [1, 1], [1, 1], [0, 0]])

        image_ = tf.pad(image_, [[0, 0], [1, 1], [1, 1], [0, 0]])
        image_ = image_ * mask + image * (1 - mask)

        image = blend(image, image_, factor)

        return convert(image, dtype, saturate=True)
