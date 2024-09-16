import numpy as np
from keras.src import backend
from keras.src import ops

from segme.utils.common.augs.common import apply
from segme.utils.common.augs.common import unwrap
from segme.utils.common.augs.common import validate
from segme.utils.common.augs.common import wrap


def rotate(image, masks, weight, prob, degrees, replace=None, name=None):
    with backend.name_scope(name or "rotate"):
        return apply(
            image,
            masks,
            weight,
            prob,
            lambda x: _rotate(x, degrees, "bilinear", replace),
            lambda x: _rotate(
                x, degrees, "nearest", replace=np.zeros([1, 1, 1, x.shape[-1]])
            ),
            lambda x: _rotate(
                x, degrees, "nearest", replace=np.zeros([1, 1, 1, x.shape[-1]])
            ),
        )


def rotate_cw(image, masks, weight, prob, name=None):
    with backend.name_scope(name or "rotate_cw"):
        return apply(
            image,
            masks,
            weight,
            prob,
            lambda x: _rotate_cw(x),
            lambda x: _rotate_cw(x),
            lambda x: _rotate_cw(x),
        )


def rotate_ccw(image, masks, weight, prob, name=None):
    with backend.name_scope(name or "rotate_ccw"):
        return apply(
            image,
            masks,
            weight,
            prob,
            lambda x: _rotate_ccw(x),
            lambda x: _rotate_ccw(x),
            lambda x: _rotate_ccw(x),
        )


def _rotate(image, degrees, interpolation, replace=None, name=None):
    with backend.name_scope(name or "rotate_"):
        image, _, _ = validate(image, None, None)

        radians = ops.cast(-degrees * np.pi / 180.0, "float32")[None]
        height, width = ops.shape(image)[1:3]
        height = ops.cast(height, "float32")
        width = ops.cast(width, "float32")

        h_offset = (
            (width - 1)
            - (ops.cos(radians) * (width - 1) - ops.sin(radians) * (height - 1))
        ) / 2.0
        v_offset = (
            (height - 1)
            - (ops.sin(radians) * (width - 1) + ops.cos(radians) * (height - 1))
        ) / 2.0
        matrix = ops.concatenate(
            [
                ops.cos(radians)[:, None],
                -ops.sin(radians)[:, None],
                h_offset[:, None],
                ops.sin(radians)[:, None],
                ops.cos(radians)[:, None],
                v_offset[:, None],
                ops.zeros((1, 2), "float32"),
            ],
            axis=1,
        )

        image = wrap(image)
        image = ops.image.affine_transform(
            image,
            matrix,
            interpolation=interpolation,
        )
        image = unwrap(image, replace)

        return image


def _rotate_cw(image, name=None):
    with backend.name_scope(name or "rotate_cw_"):
        image, _, _ = validate(image, None, None)

        image = ops.transpose(image, [0, 2, 1, 3])
        image = ops.flip(image, axis=2)

        return image


def _rotate_ccw(image, name=None):
    with backend.name_scope(name or "rotate_ccw_"):
        image, _, _ = validate(image, None, None)

        image = ops.transpose(image, [0, 2, 1, 3])
        image = ops.flip(image, axis=1)

        return image
