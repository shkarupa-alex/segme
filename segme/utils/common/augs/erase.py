import numpy as np
import tensorflow as tf
from segme.utils.common.augs.common import apply, convert, wrap, unwrap, validate
from segme.common.shape import get_shape


def erase(image, masks, weight, prob, area, replace=None, name=None):
    with tf.name_scope(name or 'erase'):
        image, masks, weight = validate(image, masks, weight)

        if isinstance(area, float):
            area = [0., area]
        area = tf.convert_to_tensor(area, 'float32')
        area = tf.unstack(tf.sqrt(area) / 2.)

        (batch, height, width), _ = get_shape(image, axis=[0, 1, 2])
        height = tf.cast(height, 'float32')
        width = tf.cast(width, 'float32')

        hcenter = tf.random.uniform([batch, 1, 1, 1], maxval=height)
        wcenter = tf.random.uniform([batch, 1, 1, 1], maxval=width)

        hradius = tf.random.uniform([batch, 1, 1, 1], minval=area[0] * height, maxval=area[1] * height)
        wradius = tf.random.uniform([batch, 1, 1, 1], minval=area[0] * width, maxval=area[1] * width)

        hrange = tf.repeat(tf.range(height)[None, :, None, None], batch, axis=0)
        wrange = tf.repeat(tf.range(width)[None, None, :, None], batch, axis=0)

        mask = ((hrange < hcenter - hradius) | (hrange > hcenter + hradius)) | \
               ((wrange < wcenter - wradius) | (wrange > wcenter + wradius))
        mask = tf.cast(mask, 'float32')

        return apply(
            image, masks, weight, prob,
            lambda x: _erase(x, mask, replace=replace),
            tf.identity,
            lambda x: _erase(x, mask, replace=np.zeros([1, 1, 1, x.shape[-1]])))


def _erase(image, mask, replace=None, name=None):
    with tf.name_scope(name or 'erase_'):
        image, _, mask = validate(image, None, mask)

        mask = tf.cast(mask, image.dtype)

        image = wrap(image)
        image *= mask
        image = unwrap(image, replace)

        return image
