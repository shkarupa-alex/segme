import numpy as np
from keras.src import backend
from keras.src import ops

from segme.utils.common.augs.common import apply
from segme.utils.common.augs.common import unwrap
from segme.utils.common.augs.common import validate
from segme.utils.common.augs.common import wrap


def translate_x(image, masks, weight, prob, factor, replace=None, name=None):
    with backend.name_scope(name or "translate_x"):
        return apply(
            image,
            masks,
            weight,
            prob,
            lambda x: _translate_x(x, factor, "bilinear", replace),
            lambda x: _translate_x(
                x, factor, "nearest", replace=np.zeros([1, 1, 1, x.shape[-1]])
            ),
            lambda x: _translate_x(
                x, factor, "nearest", replace=np.zeros([1, 1, 1, x.shape[-1]])
            ),
        )


def translate_y(image, masks, weight, prob, factor, replace=None, name=None):
    with backend.name_scope(name or "translate_y"):
        return apply(
            image,
            masks,
            weight,
            prob,
            lambda x: _translate_y(x, factor, "bilinear", replace),
            lambda x: _translate_y(
                x, factor, "nearest", replace=np.zeros([1, 1, 1, x.shape[-1]])
            ),
            lambda x: _translate_y(
                x, factor, "nearest", replace=np.zeros([1, 1, 1, x.shape[-1]])
            ),
        )


def _translate_x(image, factor, interpolation, replace=None, name=None):
    with backend.name_scope(name or "translate_x_"):
        image, _, _ = validate(image, None, None)

        width = ops.cast(ops.shape(image)[2], "float32")
        translation = ops.stack([width * factor, 0])[None]
        matrix = ops.concatenate(
            [
                ops.ones((1, 1), "float32"),
                ops.zeros((1, 1), "float32"),
                -translation[:, 0, None],
                ops.zeros((1, 1), "float32"),
                ops.ones((1, 1), "float32"),
                -translation[:, 1, None],
                ops.zeros((1, 2), "float32"),
            ],
            axis=1,
        )

        image = wrap(image)
        image = ops.image.affine_transform(
            image, matrix, interpolation=interpolation
        )
        image = unwrap(image, replace)

        return image


def _translate_y(image, factor, interpolation, replace=None, name=None):
    with backend.name_scope(name or "translate_y_"):
        image, _, _ = validate(image, None, None)

        height = ops.cast(ops.shape(image)[1], "float32")
        translation = ops.stack([0, height * factor])[None]
        matrix = ops.concatenate(
            [
                ops.ones((1, 1), "float32"),
                ops.zeros((1, 1), "float32"),
                -translation[:, 0, None],
                ops.zeros((1, 1), "float32"),
                ops.ones((1, 1), "float32"),
                -translation[:, 1, None],
                ops.zeros((1, 2), "float32"),
            ],
            axis=1,
        )

        image = wrap(image)
        image = ops.image.affine_transform(
            image, matrix, interpolation=interpolation
        )
        image = unwrap(image, replace)

        return image
