import tensorflow as tf
from segme.utils.common.augs.common import apply, validate


def flip_ud(image, masks, weight, prob, name=None):
    with tf.name_scope(name or 'flip_ud'):
        return apply(
            image, masks, weight, prob,
            lambda x: _flip_ud(x),
            lambda x: _flip_ud(x),
            lambda x: _flip_ud(x))


def flip_lr(image, masks, weight, prob, name=None):
    with tf.name_scope(name or 'flip_lr'):
        return apply(
            image, masks, weight, prob,
            lambda x: _flip_lr(x),
            lambda x: _flip_lr(x),
            lambda x: _flip_lr(x))


def _flip_ud(image, name=None):
    with tf.name_scope(name or 'flip_ud_'):
        image, _, _ = validate(image, None, None)

        image = tf.image.flip_up_down(image)

        return image


def _flip_lr(image, name=None):
    with tf.name_scope(name or 'flip_lr_'):
        image, _, _ = validate(image, None, None)

        image = tf.image.flip_left_right(image)

        return image
