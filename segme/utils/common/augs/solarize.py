import numpy as np
from keras.src import backend
from keras.src import ops

from segme.utils.common.augs.common import apply
from segme.utils.common.augs.common import validate


def solarize(image, masks, weight, prob, threshold=None, name=None):
    with backend.name_scope(name or "solarize"):
        return apply(
            image,
            masks,
            weight,
            prob,
            lambda x: _solarize(x, threshold),
            None,
            None,
        )


def _solarize(image, threshold=None, name=None):
    with backend.name_scope(name or "solarize_"):
        image, _, _ = validate(image, None, None)

        if backend.is_float_dtype(image.dtype):
            max_val = 1.0
            if threshold is None:
                threshold = 128 / 255
        else:
            dtype = backend.standardize_dtype(image.dtype)
            max_val = np.iinfo(dtype).max
            if threshold is None:
                threshold = np.round(max_val / 2).astype(dtype)

        image = ops.where(image < threshold, image, max_val - image)

        return image
