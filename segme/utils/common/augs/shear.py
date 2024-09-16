import numpy as np
from keras.src import backend
from keras.src import ops

from segme.utils.common.augs.common import apply
from segme.utils.common.augs.common import unwrap
from segme.utils.common.augs.common import validate
from segme.utils.common.augs.common import wrap


def shear_x(image, masks, weight, prob, factor, replace=None, name=None):
    with backend.name_scope(name or "shear_x"):
        return apply(
            image,
            masks,
            weight,
            prob,
            lambda x: _shear_x(x, factor, "bilinear", replace),
            lambda x: _shear_x(
                x, factor, "nearest", replace=np.zeros([1, 1, 1, x.shape[-1]])
            ),
            lambda x: _shear_x(
                x, factor, "nearest", replace=np.zeros([1, 1, 1, x.shape[-1]])
            ),
        )


def shear_y(image, masks, weight, prob, factor, replace=None, name=None):
    with backend.name_scope(name or "shear_y"):
        return apply(
            image,
            masks,
            weight,
            prob,
            lambda x: _shear_y(x, factor, "bilinear", replace),
            lambda x: _shear_y(
                x, factor, "nearest", replace=np.zeros([1, 1, 1, x.shape[-1]])
            ),
            lambda x: _shear_y(
                x, factor, "nearest", replace=np.zeros([1, 1, 1, x.shape[-1]])
            ),
        )


def _shear_x(image, factor, interpolation, replace=None, name=None):
    with backend.name_scope(name or "shear_x_"):
        image, _, _ = validate(image, None, None)

        matrix = ops.stack([1.0, factor, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])[None]

        image = wrap(image)
        image = ops.image.affine_transform(
            image, matrix, interpolation=interpolation
        )
        image = unwrap(image, replace)

        return image


def _shear_y(image, factor, interpolation, replace=None, name=None):
    with backend.name_scope(name or "shear_y_"):
        image, _, _ = validate(image, None, None)

        matrix = ops.stack([1.0, 0.0, 0.0, factor, 1.0, 0.0, 0.0, 0.0])[None]

        image = wrap(image)
        image = ops.image.affine_transform(
            image, matrix, interpolation=interpolation
        )
        image = unwrap(image, replace)

        return image
