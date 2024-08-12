import numpy as np
import tensorflow as tf

from segme.common.shape import get_shape
from segme.utils.common.augs.common import apply
from segme.utils.common.augs.common import validate


def erase(image, masks, weight, prob, area, replace=None, name=None):
    with tf.name_scope(name or "erase"):
        image, masks, weight = validate(image, masks, weight)

        if isinstance(area, float):
            area = [0.0, area]
        area = tf.convert_to_tensor(area, "float32")
        area = tf.unstack(tf.sqrt(area) / 2.0)

        (batch, height, width), _ = get_shape(image, axis=[0, 1, 2])
        height = tf.cast(height, "float32")
        width = tf.cast(width, "float32")

        hcenter = tf.random.uniform([batch, 1, 1, 1], maxval=height)
        wcenter = tf.random.uniform([batch, 1, 1, 1], maxval=width)

        hradius = tf.random.uniform(
            [batch, 1, 1, 1], minval=area[0] * height, maxval=area[1] * height
        )
        wradius = tf.random.uniform(
            [batch, 1, 1, 1], minval=area[0] * width, maxval=area[1] * width
        )

        hrange = tf.repeat(tf.range(height)[None, :, None, None], batch, axis=0)
        wrange = tf.repeat(tf.range(width)[None, None, :, None], batch, axis=0)

        mask = ((hrange < hcenter - hradius) | (hrange > hcenter + hradius)) | (
            (wrange < wcenter - wradius) | (wrange > wcenter + wradius)
        )

        return apply(
            image,
            masks,
            weight,
            prob,
            lambda x: _erase(x, mask, replace=replace),
            tf.identity,
            lambda x: _erase(x, mask, replace=np.zeros([1, 1, 1, x.shape[-1]])),
        )


def _erase(image, mask, replace=None, name=None):
    with tf.name_scope(name or "erase_"):
        image, _, mask = validate(image, None, mask)

        (batch,), _ = get_shape(image, axis=[0])
        mask = mask[:batch]

        if replace is not None:
            replace = tf.convert_to_tensor(replace, image.dtype, name="replace")
            replace, _, _ = validate(replace, None, None)
        else:
            replace = tf.reduce_mean(image, axis=[1, 2], keepdims=True)

        image = tf.where(mask, image, replace)

        return image
