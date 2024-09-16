import numpy as np
from keras.src import backend

from segme.utils.common.augs.common import apply
from segme.utils.common.augs.common import validate


def invert(image, masks, weight, prob, name=None):
    with backend.name_scope(name or "invert"):
        return apply(image, masks, weight, prob, _invert, None, None)


def _invert(image, name=None):
    with backend.name_scope(name or "invert_"):
        image, _, _ = validate(image, None, None)

        if backend.is_float_dtype(image.dtype):
            max_val = 1.0
        else:
            dtype = backend.standardize_dtype(image.dtype)
            max_val = np.iinfo(dtype).max

        image = max_val - image

        return image
