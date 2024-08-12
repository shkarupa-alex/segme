import numpy as np
import tensorflow as tf

from segme.utils.common.augs.common import apply
from segme.utils.common.augs.common import validate


def solarize(image, masks, weight, prob, threshold=None, name=None):
    with tf.name_scope(name or "solarize"):
        return apply(
            image,
            masks,
            weight,
            prob,
            lambda x: _solarize(x, threshold),
            tf.identity,
            tf.identity,
        )


def _solarize(image, threshold=None, name=None):
    with tf.name_scope(name or "solarize_"):
        image, _, _ = validate(image, None, None)

        dtype = image.dtype
        if dtype.is_floating:
            max_val = 1.0
            if threshold is None:
                threshold = 128 / 255
        else:
            max_val = dtype.max
            if threshold is None:
                threshold = np.round(dtype.max / 2).astype(
                    dtype.as_numpy_dtype()
                )

        image = tf.where(image < threshold, image, max_val - image)

        return image
